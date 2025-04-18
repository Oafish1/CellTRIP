from collections import defaultdict
import io
import os
import warnings

import numpy as np
import ray.util.collective as col  # Maybe conditional import?
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F

from . import utility as _utility


class ResidualSA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        activation=nn.ReLU(),
        num_mlps=1,
        batch_first=True,
        **kwargs
    ):
        super().__init__()

        # Parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.activation = activation
        self.num_mlps = num_mlps
        self.batch_first = batch_first

        # Attention
        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=self.batch_first, **kwargs)

        # Layer norms
        self.layer_norms = nn.ModuleList([ nn.LayerNorm(self.embed_dim) for _ in range(2+self.num_mlps) ])

        # MLP
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                self.activation,
                nn.Linear(self.embed_dim, self.embed_dim))
            for _ in range(self.num_mlps)])

    def forward(self, x):
        # Parameters
        x1 = x
        layer_norm_idx = 0

        # Apply residual self attention
        x2 = self.layer_norms[layer_norm_idx](x1)
        # x2 = x1
        layer_norm_idx += 1
        x3, _ = self.attention(x2, x2, x2)
        x1 = x1 + x3

        # Apply residual mlps
        for mlp in self.mlps:
            x2 = self.layer_norms[layer_norm_idx](x1)
            # x2 = x1
            layer_norm_idx += 1
            x3 = mlp(x2)
            x1 = x1 + x3

        # Final layer norm
        xf = self.layer_norms[layer_norm_idx](x1)
        # xf = x1

        return xf


### Policy classes
class EntitySelfAttention(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        feature_embed_dim=32,
        embed_dim=64,
        num_heads=4,
        action_std_init=.6,
        activation=nn.ReLU(),
        final_activation=nn.Tanh(),
        num_mlps=1,
        critic=False,
        **kwargs,
    ):
        super().__init__()

        # Base information
        self.positional_dim = positional_dim
        self.modal_dims = modal_dims
        self.output_dim = output_dim
        self.feature_embed_dim = feature_embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.activation = activation
        self.final_activation = final_activation
        self.num_mlps = num_mlps
        self.critic = critic

        # Set action std
        self.action_std = nn.Parameter(torch.tensor(action_std_init))

        # Feature embedding
        self.feature_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.modal_dims[i], self.feature_embed_dim),
                self.activation,
                nn.Linear(self.feature_embed_dim, self.feature_embed_dim),
                nn.LayerNorm(self.feature_embed_dim))
            for i in range(len(self.modal_dims))])

        # Embedding
        solo_features_len = self.feature_embed_dim * len(self.modal_dims) + self.positional_dim
        self.self_embed = nn.Sequential(
            nn.Linear(solo_features_len, self.embed_dim),
            self.activation,
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim))
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Sequential(
            nn.Linear(self.embed_dim + solo_features_len, self.embed_dim),
            self.activation,
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim))  # Not across entities

        # Self attention
        self.residual_self_attention = ResidualSA(self.embed_dim, self.num_heads, activation=self.activation, num_mlps=self.num_mlps)

        # Decision
        self.decider = nn.Sequential(
            nn.Linear(2*self.embed_dim, 2*self.embed_dim),
            self.activation,
            nn.Linear(2*self.embed_dim, self.output_dim))

    ### Calculation functions
    def embed_features(self, entities):
        # TODO: Maybe there's a more efficient way to do this with masking
        running_idx = self.positional_dim
        ret = [entities[..., :running_idx]]
        for ms, fe in zip(
            self.modal_dims,
            self.feature_embed):
            # Record embedded features
            # Maybe pre-ln if we don't care about magnitude
            val = fe(entities[..., running_idx:(running_idx + ms)])
            val = self.activation(val)

            # Record
            ret.append(val)
            running_idx += ms # Increment start idx

        # Check shape
        assert running_idx == entities.shape[-1]

        # Construct full matrix
        ret = torch.concat(ret, dim=-1)

        return ret

    def calculate_actions(self, state):
        # Formatting
        self_entity, node_entities = state

        # Feature embedding (could be done with less redundancy)
        self_entity = self.embed_features(self_entity)
        node_entities = self.embed_features(node_entities)

        # Self embedding
        self_embed = self.self_embed(self_entity).unsqueeze(-2)  # This has blown up with multi-gpu backward

        # Node embeddings
        node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_entities.shape[:-1], self_embed.shape[-1]), node_entities), dim=-1))

        # Self attention across entities
        embeddings = torch.concat((self_embed, node_embeds), dim=-2)
        attentions = self.residual_self_attention(embeddings)
        attentions_pool = attentions.mean(dim=-2)  # Average across entities
        embedding = torch.concat((self_embed.squeeze(-2), attentions_pool), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)

        # Decision
        # TODO (Minor): Should layer norm be added here and for feature embedding? Probably not.
        actions = self.decider(embedding)
        if self.final_activation is not None: actions = self.final_activation(actions)

        return actions

    def evaluate_state(self, state):
        return self.calculate_actions(state).squeeze(-1)

    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        dist = MultivariateNormal(
            loc=actions,
            covariance_matrix=torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0),
            # scale_tril=torch.linalg.cholesky(covariance_matrix),  # Speeds up computation compared to using cov matrix
            validate_args=False,  # Speeds up computation
        )

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action)

        # Return
        ret = ()
        if not set_action:
            ret += (action,)
        ret += (action_log,)
        if return_entropy: ret += (dist.entropy(),)
        if len(ret) == 1: ret = ret[0]
        return ret

    def evaluate_action(self, state, action):
        actions = self.calculate_actions(state)
        action_log, dist_entropy = self.select_action(actions, action=action, return_entropy=True)

        return action_log, dist_entropy


### Training classes
class PPO(nn.Module):
    def __init__(
            self,
            positional_dim,
            modal_dims,
            output_dim,
            # Model Parameters
            model=EntitySelfAttention,
            # Forward
            action_std=.6,
            forward_batch_size=int(5e4),
            vision_size=int(1e2),
            sample_strategy='random-proximity',
            sample_dim=None,
            reproducible_strategy='mean',
            # Weights
            epsilon_ppo=torch.inf,  # Formerly .2,
            epsilon_critic=torch.inf,  # Formerly .1,
            critic_weight=.5,
            entropy_weight=1e-3,  # Formerly 1e-2
            kl_beta_init=1.,
            kl_beta_increment=(.5, 2),
            kl_target=.1,
            kl_early_stop=False,
            # Optimizers
            action_std_lr=1e-2,
            actor_lr=3e-4,
            critic_lr=1e-3,
            weight_decay=0,  # 1e-3,
            betas=(.9, .999),  # (.997, .997),
            lr_gamma=.95,
            # Backward
            update_iterations=10,  # Can also be 'auto'
            sync_iterations=1,
            pool_size=None,
            epoch_size=None,
            batch_size=int(1e4),
            minibatch_size=int(1e4),
            load_level='batch',
            cast_level='minibatch',
            device='cpu',
            **kwargs,
    ):
        super().__init__()

        # Parameters
        self.positional_dim = positional_dim
        self.modal_dims = modal_dims.copy()
        self.output_dim = output_dim

        # Variables
        self.epsilon_ppo = epsilon_ppo
        self.epsilon_critic = epsilon_critic
        self.kl_beta = nn.Parameter(torch.tensor(kl_beta_init), requires_grad=False)
        self.kl_beta_increment = kl_beta_increment
        self.kl_target = kl_target
        self.kl_early_stop = kl_early_stop

        # Runtime management
        self.forward_batch_size = forward_batch_size
        # NOTE: Assumes output corresponds to positional dims if not explicitly provided
        if sample_dim is None: sample_dim = output_dim
        self.split_args = {
            'max_nodes': vision_size,
            'sample_strategy': sample_strategy,
            'reproducible_strategy': reproducible_strategy,
            'sample_dim': sample_dim,
        }
        self.update_iterations = update_iterations
        self.sync_iterations = sync_iterations
        self.pool_size = pool_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.load_level = load_level
        self.cast_level = cast_level
        self.device = device

        # New policy
        self.actor = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=output_dim, action_std=action_std, **kwargs)
        self.critic = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=1, final_activation=None, **kwargs)
        self.optimizer = torch.optim.Adam([  # CHANGED
            {'params': list(filter(lambda kv: kv[0] in ('action_std',), self.actor.named_parameters())), 'lr': action_std_lr},
            {'params': list(filter(lambda kv: kv[0] not in ('action_std',), self.actor.named_parameters())), 'lr': actor_lr},
            {'params': self.critic.named_parameters(), 'lr': critic_lr}],
            betas=betas, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma, verbose=True)

        # Old policy
        self.actor_old = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=output_dim, action_std=action_std, **kwargs)
        self.critic_old = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=1, final_activation=None, **kwargs)
        self.optimizer_old = torch.optim.Adam([
            {'params': list(filter(lambda kv: kv[0] in ('action_std',), self.actor_old.named_parameters())), 'lr': action_std_lr},
            {'params': list(filter(lambda kv: kv[0] not in ('action_std',), self.actor_old.named_parameters())), 'lr': actor_lr},
            {'params': self.critic_old.named_parameters(), 'lr': critic_lr}],
            betas=betas, weight_decay=weight_decay)
        self.copy_policy()

        # Iteration tracking
        self.policy_iteration = nn.Parameter(torch.tensor(0.), requires_grad=False)

        # Weights
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

        # To device
        self.to(self.device)

    ### Minor functions
    def to(self, device):
        ret = super().to(device)
        self.device = self.policy_iteration.get_device()
        return ret

    def get_action_std(self):
        return self.actor.action_std.detach().item()

    def get_policy_iteration(self):
        return int(self.policy_iteration.item())
    
    def copy_policy(self, _invert=False):
        "Copy new policy weights onto old policy"
        sources, targets = (
            (self.actor, self.critic, self.optimizer),
            (self.actor_old, self.critic_old, self.optimizer_old))
        if _invert: sources, targets = targets, sources
        for source, target in zip(sources, targets):
            target.load_state_dict(source.state_dict())

    def revert_policy(self):
        "Copy old policy weights onto new policy"
        self.copy_policy(_invert=True)

    ### Saving functions
    def save_checkpoint(self, directory, name=None):
        # Defaults
        name = 'celltrip' if name is None else name

        # Get all vars in order
        fname = os.path.join(directory, f'{name}-{int(self.policy_iteration.item()):0>4}.weights')
        policy_state = _utility.general.get_policy_state(self)

        # Save
        if fname.startswith('s3://'):
            # Get s3 handler
            s3 = _utility.general.get_s3_handler_with_access(fname)

            # Get buffer
            # buffer = io.BytesIO()
            # torch.save(policy_state, buffer)

            # Save
            with s3.open(fname, 'wb') as f:
                torch.save(policy_state, f)
        else:
            os.makedirs(directory, exist_ok=True)
            torch.save(policy_state, fname)

        return fname

    def load_checkpoint(self, fname):
        # Get from fname
        if fname.startswith('s3://'):
            # Get s3 handler
            s3 = _utility.general.get_s3_handler_with_access(fname)

            # Retrieve object from store
            handle = s3.open(fname, 'rb')  # NOTE: Never closed
            policy_state = torch.load(handle, map_location=self.device)
        else:
            policy_state = torch.load(fname, map_location=self.device)

        # Load policy
        _utility.general.set_policy_state(self, policy_state)

    ### Running functions
    def act(self, *state, return_all=False):
        # Add dimension if only one shape
        if len(state[0].shape) == 1:
            state = [s.unsqueeze[0] for s in state]

        # Calculate actions and state
        actions = self.actor.calculate_actions(state)
        action, action_log = self.actor.select_action(actions)
        state_val = self.critic.evaluate_state(state)

        if return_all: return action, action_log, state_val
        return action

    def act_macro(self, state, *, keys=None, memory=None, forward_batch_size=None):
        # Data Checks
        assert state.shape[0] > 0, 'Empty state matrix passed'
        if keys is not None: assert len(keys) == state.shape[0], (
            f'Length of keys vector must equal state dimension 0 ({state.shape[0]}), '
            f'got {len(keys)} instead.'
        )
            
        # Defaults
        if forward_batch_size is None: forward_batch_size = self.forward_batch_size

        # Act
        if forward_batch_size is not None:
            # Compute `max_batch` at a time with randomized `max_nodes`
            initialized = False
            for start_idx in range(0, state.shape[0], forward_batch_size):
                state_split = _utility.processing.split_state(
                    state,
                    idx=list( range(start_idx, min(start_idx+forward_batch_size, state.shape[0])) ),
                    **self.split_args)
                action_sub, action_log_sub, state_val_sub = self.act(*state_split, return_all=True)

                # Concat
                if not initialized:
                    action, action_log, state_val = action_sub, action_log_sub, state_val_sub
                    initialized = True
                else:
                    action = torch.concat((action, action_sub), dim=0)
                    action_log = torch.concat((action_log, action_log_sub), dim=0)
                    state_val = torch.concat((state_val, state_val_sub), dim=0)
        else:
            # Compute all at once
            action, action_log, state_val = self.act(*_utility.processing.split_state(state, **self.split_args), return_all=True)
        
        # Record
        # NOTE: `reward` and `is_terminal` are added outside of the class, calculated
        # after stepping the environment
        if memory is not None and keys is not None and self.training:
            memory.record(
                keys=keys,
                states=state,
                actions=action,
                action_logs=action_log,
                state_vals=state_val)

        return action

    def forward(self, *state):
        # Calculate action
        actions = self.actor.calculate_actions(state)
        action, _ = self.actor.select_action(actions)

        return action

    ### Backward functions
    def update(
        self,
        memory,
        update_iterations=None,
        verbose=False,
        # Collective args
        sync_iterations=None,
    ):
        # NOTE: The number of epochs is spread across `world_size` workers
        # NOTE: Assumes col.init_collective_group has already been called if world_size > 1
        # Parameters
        if update_iterations is None: update_iterations = self.update_iterations
        if sync_iterations is None: sync_iterations = self.sync_iterations

        # Collective operations
        use_collective = col.is_group_initialized('default')

        # Batch parameters
        level_dict = {'pool': 0, 'epoch': 1, 'batch': 2, 'minibatch': 3}
        load_level = level_dict[self.load_level]
        cast_level = level_dict[self.cast_level]
        assert cast_level >= load_level, 'Cannot cast without first loading'

        # Determine level sizes
        memory_size = len(memory)
        pool_size = self.pool_size if self.pool_size is not None else memory_size
        pool_size = int(min(pool_size, memory_size))
        epoch_size = self.epoch_size if self.epoch_size is not None else pool_size
        epoch_size = int(min(epoch_size, pool_size))
        batch_size = self.batch_size if self.batch_size is not None else epoch_size
        batch_size = int(min(batch_size, epoch_size))
        minibatch_size = self.minibatch_size if self.minibatch_size is not None else batch_size
        minibatch_size = int(min(minibatch_size, batch_size))

        # Get number of iterations
        if update_iterations == 'auto': update_iterations = .5 * memory.get_new_len() // batch_size  # Scale by memory size
        # world_size = 1 if not use_collective else col.get_collective_group_size()
        # update_iterations = np.ceil(update_iterations/world_size).astype(int)  # Scale by num workers

        # Cap at max size to reduce redundancy for sequential samples
        # NOTE: Pool->epoch is the only non-sequential sample, and is thus not included here
        max_unique_memories = batch_size * update_iterations
        epoch_size = min(epoch_size, max_unique_memories)

        # Load pool
        pool_losses = defaultdict(lambda: 0)
        pool_data = _utility.processing.sample_and_cast(
            memory, None, None, pool_size,
            current_level=0, load_level=load_level, cast_level=cast_level,
            device=self.device)

        # Train
        iterations = 0; synchronized = True; escape = False
        while True:
            # Load epoch
            epoch_data = _utility.processing.sample_and_cast(
                memory, pool_data, pool_size, epoch_size,
                current_level=1, load_level=load_level, cast_level=cast_level,
                device=self.device)
            batches = np.ceil(epoch_size/batch_size).astype(int) if epoch_data is not None else 1
            for batch_num in range(batches):
                # Load batch
                batch_losses = defaultdict(lambda: 0)
                batch_data = _utility.processing.sample_and_cast(
                    memory, epoch_data, epoch_size, batch_size,
                    current_level=2, load_level=load_level, cast_level=cast_level,
                    device=self.device, sequential_num=batch_num)
                minibatches = np.ceil(batch_size/minibatch_size).astype(int) if batch_data is not None else 1
                for minibatch_num in range(minibatches):
                    # Load minibatch
                    minibatch_data = _utility.processing.sample_and_cast(
                        memory, batch_data, batch_size, minibatch_size,
                        current_level=3, load_level=load_level, cast_level=cast_level,
                        device=self.device, sequential_num=minibatch_num)

                    # Get subset data
                    states = minibatch_data['states']
                    actions = minibatch_data['actions']
                    action_logs = minibatch_data['action_logs']
                    state_vals = minibatch_data['state_vals']
                    advantages = minibatch_data['advantages']

                    # Perform backward
                    loss, loss_ppo, loss_critic, loss_entropy, loss_kl = self.backward(
                        states, actions, action_logs, state_vals, advantages)

                    # Scale and calculate gradient
                    accumulation_frac = states[0].shape[0] / batch_size
                    loss = loss * accumulation_frac
                    loss.backward()  # Longest computation

                    # Scale and record
                    batch_losses['Total'] += (loss.detach() * accumulation_frac).item()
                    batch_losses['PPO'] += (loss_ppo.detach().mean() * accumulation_frac).item()
                    batch_losses['critic'] += (loss_critic.detach().mean() * accumulation_frac).item()
                    batch_losses['entropy'] += (loss_entropy.detach().mean() * accumulation_frac).item()
                    batch_losses['KL'] += (loss_kl.detach().mean() * accumulation_frac).item()

                # Escape and roll back if KLD too high
                if self.kl_early_stop:
                    loss_kl = torch.tensor(batch_losses['KL'], device=self.device)
                    self.synchronize('learners', sync_list=[loss_kl])
                    if loss_kl >= self.kl_target:
                        if iterations - sync_iterations > 0:
                            # Revert to previous synchronized state within kl target
                            self.revert_policy()
                            # for w in self.actor.state_dict().values(): assert w.grad is None  # Just in case
                            iterations -= sync_iterations
                            escape = True; break
                        else:
                            warnings.warn(
                                'Update exceeded KL target too fast! Proceeding with update, but may be unstable. '
                                'Try lowering clip or learning rate parameters.')
                            escape = True; break

                # Synchronize GPU policies
                # NOTE: Synchronize gradients every batch if =1, else synchronize whole model
                # NOTE: =1 keeps optimizers in sync without need for whole-model synchronization
                if sync_iterations == 1: self.synchronize('learners', grad=True)  # Sync only grad

                # Step
                if self.kl_early_stop and synchronized: self.copy_policy()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Synchronize GPU policies
                if sync_iterations != 1:
                    sync_loop = (iterations) % sync_iterations == 0
                    last_epoch = iterations == update_iterations
                    if use_collective and (sync_loop or last_epoch):
                        self.synchronize('learners')
                        synchronized = True
                    else: synchronized = False
                # if iterations == 9: print(self.state_dict())  # Check that weights are the same across nodes

                # Increment
                iterations += 1

                # CLI
                if verbose and ((iterations) % 10 == 0 or iterations in (1, 5)):
                    print(
                        f'Iteration {iterations:02} - '
                        ' + '.join([f'{k} ({v:.3f})' for k, v in batch_losses.items()]),
                        f' :: Action STD ({self.get_action_std():.3f})')
                    
                # Record
                for k, v in batch_losses.items(): pool_losses[k] += v
                    
                # Break
                if iterations >= update_iterations: escape = True
                if escape: break

            # Break
            if escape: break

        # Update scheduler
        self.scheduler.step()

        # Update records
        self.policy_iteration += 1
        self.copy_policy()
        for k in pool_losses: pool_losses[k] /= iterations

        # Return self
        return iterations, dict(batch_losses)  # dict(pool_losses)

    def synchronize(self, group='default', sync_list=None, grad=False, broadcast=None, allreduce=None):
        # Defaults
        if broadcast is None: broadcast = False
        if allreduce is None: allreduce = not broadcast

        # Collective operations
        try: world_size = col.get_collective_group_size(group)
        except:
            warnings.warn(f'Synchronize called but no group "{group}" found.')
            return
        # rank = col.get_rank(group)

        # Sync
        sync_list = self.parameters() if sync_list is None else sync_list
        with torch.no_grad():
            for w in sync_list:  # zip(self.state_dict(), self.parameters())
                if grad: w = w.grad  # No in-place modification here
                if w is None: continue
                if broadcast: col.broadcast(w, 0, group)
                if allreduce:
                    col.allreduce(w, group)
                    w /= world_size

    def backward(
        self,
        states,
        actions,
        action_logs,
        state_vals,
        advantages):
        # Get normalized rewards and inferred rewards
        advantages_mean, advantages_std = advantages.mean(), advantages.std()
        # print(f'{self.get_policy_iteration()} - {advantages_mean:.3f} - {advantages_std:.3f}')
        normalized_advantages = (advantages - advantages_mean) / advantages_std
        inferred_rewards = advantages + state_vals

        # Evaluate actions and states
        action_logs_new, dist_entropy = self.actor.evaluate_action(states, actions)
        state_vals_new = self.critic.evaluate_state(states)
        
        # Calculate PPO loss
        ratios = torch.exp(action_logs_new - action_logs)
        unclipped_ppo = ratios * normalized_advantages
        clipped_ppo = torch.clamp(ratios, 1-self.epsilon_ppo, 1+self.epsilon_ppo) * normalized_advantages
        loss_ppo = -torch.min(unclipped_ppo, clipped_ppo)

        # Calculate critic loss
        unclipped_critic = (state_vals_new - inferred_rewards).square()
        clipped_state_vals_new = torch.clamp(state_vals_new, state_vals-self.epsilon_critic, state_vals+self.epsilon_critic)
        clipped_critic = (clipped_state_vals_new - inferred_rewards).square()
        loss_critic = torch.max(unclipped_critic, clipped_critic)

        # Calculate entropy bonus
        # NOTE: Not included in training grad if action_std is constant
        loss_entropy = -dist_entropy

        # Calculate KL divergence (approximate pointwise KL)
        # NOTE: A bit odd when it comes to replay
        loss_kl = (action_logs - action_logs_new)  # * action_logs.exp()

        # Mask and scale where needed
        # loss_kl[~new_memories] = 0
        # loss_kl = loss_kl * loss_kl.shape[0] / new_memories.sum()

        # Construct final loss
        loss = (
            loss_ppo
            + self.critic_weight * loss_critic
            + self.entropy_weight * loss_entropy
            + self.kl_beta * loss_kl)
        loss = loss.mean()

        # Update KL beta
        # NOTE: Same as Torch KLPENPPOLoss implementation
        if loss_kl.mean() < self.kl_target / 1.5: self.kl_beta.data *= self.kl_beta_increment[0]
        elif loss_kl.mean() > self.kl_target * 1.5: self.kl_beta.data *= self.kl_beta_increment[1]

        return loss, loss_ppo, loss_critic, loss_entropy, loss_kl
