import numpy as np
import ray.util.collective as col  # Maybe conditional import?
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F

from . import decorator as _decorator
from . import utility as _utility


class ResidualSA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        activation=F.tanh,
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

        # MLP
        self.mlps = nn.ModuleList([ nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.num_mlps) ])

    def forward(self, x):
        # Apply self attention
        attention, _ = self.attention(x, x, x)
        if self.num_mlps == 0: return attention

        # Apply first residual mlp
        activations = self.mlps[0](attention)
        activations = self.activation(activations)
        x = x + activations

        # Apply further residual mlps
        for i in range(1, self.num_mlps):
            activations = self.activation(self.mlps[i](x))
            x = x + activations

        return x


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
        activation=F.tanh,
        num_mlps=1,
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
        self.num_mlps = num_mlps

        # Set action std
        self.action_var = nn.Parameter(torch.full((self.output_dim,), 0.), requires_grad=False)
        self.scale_tril = nn.Parameter(torch.zeros((self.output_dim, self.output_dim)), requires_grad=False)
        self.set_action_std(action_std_init)

        # Layer normalization
        self.layer_norm = nn.ModuleDict({
            'self embedding': nn.LayerNorm(self.embed_dim),
            'node embedding': nn.LayerNorm(self.embed_dim),  # Not across entities
            'residual self attention': nn.LayerNorm(self.embed_dim),
        })

        # Feature embedding
        self.feature_embed = nn.ModuleList([ nn.Linear(self.modal_dims[i], self.feature_embed_dim) for i in range(len(self.modal_dims)) ])

        # Embedding
        solo_features_len = self.feature_embed_dim * len(self.modal_dims) + self.positional_dim
        self.self_embed = nn.Linear(solo_features_len, self.embed_dim)
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Linear(self.embed_dim + solo_features_len, self.embed_dim)

        # Self attention
        self.residual_self_attention = ResidualSA(self.embed_dim, self.num_heads, activation=self.activation, num_mlps=self.num_mlps)

        # Decision
        self.decider = nn.Linear(2*self.embed_dim, self.output_dim)

    ### Training functions
    def set_action_std(self, new_action_std):
        self.action_var.fill_(new_action_std**2)  # Spent like a day+.5 trying to debug, realized I forgot **2
        covariance = torch.diag(self.action_var).unsqueeze(dim=0)
        self.scale_tril.data = torch.linalg.cholesky(covariance)  # Generally faster on CPU

    ### Calculation functions
    def embed_features(self, entities):
        running_idx = self.positional_dim
        ret = [entities[..., :running_idx]]
        for ms, fe in zip(self.modal_dims, self.feature_embed):
            # Record embedded features
            val = fe(entities[..., running_idx:(running_idx + ms)])
            val = self.activation(val)
            # val = entities[..., running_idx:(running_idx + ms)][..., :self.feature_embed_dim]  # TEST
            ret.append(val)

            # Increment start idx
            running_idx += ms

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
        self_embed = self.layer_norm['self embedding'](self_embed)
        self_embed = self.activation(self_embed)

        # Node embeddings
        node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_entities.shape[:-1], self_embed.shape[-1]), node_entities), dim=-1))
        node_embeds = self.layer_norm['node embedding'](node_embeds)
        node_embeds = self.activation(node_embeds)

        # Self attention across entities
        embeddings = torch.concat((self_embed, node_embeds), dim=-2)
        attentions = self.residual_self_attention(embeddings)
        attentions = self.layer_norm['residual self attention'](attentions)
        attentions_pool = attentions.mean(dim=-2)  # Average across entities
        embedding = torch.concat((self_embed.squeeze(-2), attentions_pool), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)

        # Decision
        actions = self.decider(embedding)
        # TODO (Minor): Should layer norm be added here and for feature embedding?
        # TODO (Major): Maybe remove activation here for critic?
        actions = self.activation(actions)

        return actions

    def evaluate_state(self, state):
        return self.calculate_actions(state).squeeze(-1)

    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        dist = MultivariateNormal(
            loc=actions,
            # covariance_matrix=torch.diag(self.action_var).unsqueeze(dim=0),
            scale_tril=self.scale_tril,  # Speeds up computation compared to using cov matrix
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
            action_std_init=.6,
            action_std_decay=.05,
            action_std_min=.15,
            epsilon_clip=.2,
            actor_lr=3e-4,
            critic_lr=1e-3,
            lr_gamma=.99,
            # Forward
            forward_batch_size=int(5e4),
            vision_size=int(1e2),
            sample_strategy='random-proximity',
            sample_dim=None,
            reproducible_strategy='mean',
            # Backward
            update_iterations=80,  # Can also be 'auto'
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
        self.action_std = action_std_init
        self.action_std_decay = action_std_decay
        self.action_std_min = action_std_min
        self.epsilon_clip = epsilon_clip

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
        self.pool_size = pool_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.load_level = load_level
        self.cast_level = cast_level
        self.device = device

        # New policy
        self.actor = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=output_dim, action_std_init=action_std_init, **kwargs)
        self.critic = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=1, **kwargs)

        # Old policy
        # self.actor_old = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=output_dim, action_std_init=action_std_init, **kwargs)
        # self.critic_old = model(positional_dim=positional_dim, modal_dims=modal_dims, output_dim=1, **kwargs)

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr},
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Memory
        # self.memory = AdvancedMemoryBuffer(sum(modal_dims), rs_nset=rs_nset, split_args=self.split_args)

        # Copy current weights
        # self.update_old_policy()

        # To device
        self.to(self.device)

    ### Base overloads
    def to(self, device):
        self.device = device
        return super().to(device)

    ### Utility functions
    # def update_old_policy(self):
    #     self.actor_old.load_state_dict(self.actor.state_dict())
    #     self.critic_old.load_state_dict(self.critic.state_dict())

    def decay_action_std(self):
        self.action_std = max(self.action_std - self.action_std_decay, self.action_std_min)
        self.actor.set_action_std(self.action_std)
        # self.actor_old.set_action_std(self.action_std)

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
        sync_epochs=5,
    ):
        # NOTE: The number of epochs is spread across `world_size` workers
        # NOTE: Assumes col.init_collective_group has already been called if world_size > 1
        # Parameters
        if update_iterations is None: update_iterations = self.update_iterations

        # Collective operations
        use_collective = col.is_group_initialized('default')
        world_size = 1 if not use_collective else col.get_collective_group_size()

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
        # update_iterations = np.ceil(update_iterations/world_size).astype(int)  # Scale by num workers

        # Cap at max size to reduce redundancy for sequential samples
        # NOTE: Pool->epoch is the only non-sequential sample, and is thus not included here
        max_unique_memories = batch_size * update_iterations
        epoch_size = min(epoch_size, max_unique_memories)

        # Load pool
        pool_data = _utility.processing.sample_and_cast(
            memory, None, None, pool_size,
            current_level=0, load_level=load_level, cast_level=cast_level,
            device=self.device)

        # Train
        iterations = 0
        while True:
            # Load epoch
            epoch_data = _utility.processing.sample_and_cast(
                memory, pool_data, pool_size, epoch_size,
                current_level=1, load_level=load_level, cast_level=cast_level,
                device=self.device)
            batches = np.ceil(epoch_size/batch_size).astype(int) if epoch_data is not None else 1
            for batch_num in range(batches):
                # Load batch
                batch_loss = batch_ppo = batch_critic = batch_entropy = 0
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
                    rewards = minibatch_data['normalized_rewards']

                    # Perform backward
                    loss, loss_PPO, loss_critic, loss_entropy = self.backward(
                        states, actions, action_logs, state_vals, rewards)

                    # Scale and calculate gradient
                    accumulation_frac = rewards.shape[0] / batch_size
                    loss = loss * accumulation_frac
                    loss.backward()  # Longest computation

                    # Scale and record
                    batch_loss += loss.detach().mean()
                    batch_ppo += loss_PPO.detach().mean() * accumulation_frac
                    batch_critic += loss_critic.detach().mean() * accumulation_frac
                    batch_entropy += loss_entropy.detach().mean() * accumulation_frac

                # Step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Increment
                iterations += 1
                    
                # Synchronize GPU policies
                sync_loop = (iterations) % sync_epochs == 0
                last_epoch = iterations == update_iterations
                if use_collective and (sync_loop or last_epoch):
                    self.synchronize()
                # if epoch_num == 9: print(self.state_dict())  # Check that weights are the same across nodes

                # CLI
                if verbose and ((iterations) % 10 == 0 or iterations in (1, 5)):
                    print(
                        f'Iteration {iterations:02}'
                        f' - Loss ({batch_loss.item():.3f}) ='
                        f' PPO ({batch_ppo.item():.3f}) +'
                        f' Critic ({batch_critic.item():.3f}) +'
                        f' Entropy ({batch_entropy.item():.3f})')
                    
                # Break
                if iterations >= update_iterations: break
            if iterations >= update_iterations: break

        # Update scheduler
        self.scheduler.step()

        # Return self
        return self

    def synchronize(self, group='learners'):
        # TODO: Maybe call scheduler twice if two updates are aggregated?
        # Collective operations
        use_collective = col.is_group_initialized('default')
        world_size = 1 if not use_collective else col.get_collective_group_size()

        # Sync
        # NOTE: Optimizer momentum not synced
        for k, w in self.state_dict().items():
            col.allreduce(w, group)
            w /= world_size

    def backward(
        self,
        states,
        actions,
        action_logs,
        state_vals,
        rewards,
    ):
        # Get subset rewards
        advantages = rewards - state_vals

        # Evaluate actions
        action_logs_new, dist_entropy = self.actor.evaluate_action(states, actions)

        # Evaluate states
        state_vals_new = self.critic.evaluate_state(states)

        # Ratio between new and old probabilities
        ratios = torch.exp(action_logs_new - action_logs)

        # Calculate PPO loss
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
        loss_PPO = -torch.min(unclipped, clipped)

        # Calculate critic loss
        loss_critic = .5 * F.mse_loss(state_vals_new, rewards)

        # Calculate entropy loss
        # TODO (Minor): Figure out purpose
        # NOTE: Found out, not even in the computation graph and does nothing, delete?
        loss_entropy = -.01 * dist_entropy

        # Construct final loss
        loss = loss_PPO + loss_critic + loss_entropy
        loss = loss.mean()

        return loss, loss_PPO, loss_critic, loss_entropy
