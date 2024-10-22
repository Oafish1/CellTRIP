from collections import defaultdict

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F

from . import utilities


### Utility classes
class AdvancedMemoryBuffer:
    "Memory-efficient implementation of memory"
    def __init__(self, suffix_len, rs_nset=1e5):
        # User parameters
        self.suffix_len = suffix_len

        # Storage variables
        self.storage = {
            'keys': [],             # Lists containing keys in the first dim of states
            'states': [],           # State tensors of dim `keys x non-suffix features`
            'actions': [],          # Actions
            'action_logs': [],      # Action probabilities
            'state_vals': [],       # Critic evaluation of state
            'rewards': [],          # Rewards, list of lists
            'is_terminals': [],     # Booleans indicating if the terminal state has been reached
        }
        self.persistent_storage = {
            'suffixes': {},         # Suffixes corresponding to keys
            'suffix_matrices': {},  # Orderings of suffixes
        }

        # Maintenance variables
        self.recorded = {k: False for k in self.storage}

        # Moving statistics
        self.running_statistics = utilities.RunningStatistics(n_set=rs_nset)

    def __getitem__(self, idx):
        # Parameters
        if not utilities.is_list_like(idx): idx = [idx]
        idx = np.array(idx)

        # Initialization
        ret = defaultdict(lambda: [])

        # Sort idx
        sort_idx = np.argsort(idx)
        sort_inverse_idx = np.argsort(sort_idx)

        # Search for index
        current_index = 0
        running_index = 0
        sorted_idx = idx[sort_idx]
        for list_num in range(len(self.storage['keys'])):
            list_len = len(self.storage['keys'][list_num])
            while current_index < len(idx) and running_index + list_len > sorted_idx[current_index]:
                # Useful shortcuts
                local_idx = sorted_idx[current_index] - running_index

                # Get values
                for k in self.storage:
                    # Skip certain keys
                    if k in ['keys', 'rewards', 'is_terminals']: continue

                    # Special cases
                    if k == 'states':
                        # `_append_suffix` takes most time without caching, then `split_state`
                        val = utilities.split_state(  # TIME BOTTLENECK
                            self._append_suffix(self.storage[k][list_num], keys=self.storage['keys'][list_num]),  # TIME BOTTLENECK
                            idx=local_idx,
                        )

                    # Main case
                    else:
                        val = self.storage[k][list_num][local_idx]

                    # Record
                    ret[k].append(val)

                # Iterate
                current_index += 1

            # Iterate list start
            running_index += list_len

            # Break if idx retrieved
            if current_index >= len(idx): break

        # Catch if not all found
        if current_index < len(idx): raise IndexError(f'Index {sorted_idx[current_index]} out of range')

        # Sort to indexing order and stack
        for k in ret:
            # Sort
            ret[k] = [ret[k][i] for i in sort_inverse_idx]

            # Stack
            if k == 'states':
                ret[k] = [torch.concat([s[i] for s in ret[k]], dim=0) for i in range(2)]
            else:
                ret[k] = torch.stack(ret[k], dim=0)

        return dict(ret)

    def __len__(self):
        return sum(len(keys) for keys in self.storage['keys'])

    def _append_suffix(self, state, *, keys, cache=True):
        "Append suffixes to state vector with optional cache for common key layouts"
        # Read from cache
        # NOTE: Strings from numpy arrays are slower as keys
        if cache and keys in self.persistent_storage['suffix_matrices']:
            suffix_matrix = self.persistent_storage['suffix_matrices'][keys]

        else:
            # Aggregate suffixes
            suffix_matrix = None
            for k in keys:
                val = self.persistent_storage['suffixes'][k].unsqueeze(0)
                if suffix_matrix is None: suffix_matrix = val
                else: suffix_matrix = torch.concat((suffix_matrix, val), dim=0)

            # Add to cache
            if cache: self.persistent_storage['suffix_matrices'][keys] = suffix_matrix

        # Append to state
        return torch.concat((state, suffix_matrix), dim=1)

    def _flat_index_to_index(self, idx):
        "Convert int index to grouped format that can be used on keys, state, etc."
        # Basic checks
        assert not utilities.is_list_like(idx) and idx >= 0, 'Index must be a positive integer'

        # Search for index
        running_index = 0
        for list_num in range(len(self.keys)):
            list_len = len(self.keys[list_num])
            if running_index + list_len > idx:
                return (list_num, idx - running_index)

        # Throw error if not found
        raise IndexError('Index out of range')

    def record(self, **kwargs):
        "Record passed variables"
        # Check that passed variables haven't been stored yet for this record
        for k in kwargs:
            assert k in self.storage, f'`{k}` not found in memory object'
            assert not self.recorded[k], f'`{k}` has already been recorded for this record'

        # Store new variables
        for k, v in kwargs.items():
            # Special cases
            # NOTE: For some god-forsaken reason, traeating this as an np array or non-sequential list brings unimaginable problems (2x backwards time)
            if k == 'keys': v = tuple(v)

            # Record
            self.storage[k].append(v)
            self.recorded[k] = True

        # Reset if all variables have been recorded
        if np.array([v for _, v in self.recorded.items()]).all():
            # Gleam suffixes
            for j, k in enumerate(self.storage['keys'][-1]):
                if k not in self.persistent_storage['suffixes']:
                    self.persistent_storage['suffixes'][k] = self.storage['states'][-1][j][-self.suffix_len:].clone().cpu()

            # Cut suffixes
            # Note: MUST BE CLONED otherwise stores whole unsliced tensor
            self.storage['states'][-1] = self.storage['states'][-1][..., :-self.suffix_len].clone().cpu()

            # Set all variables as unrecorded
            for k in self.recorded: self.recorded[k] = False

    def propagate_rewards(self, gamma=.95, prune=None):
        "Propagate rewards with decay"
        ret = []
        if prune is not None: ret_prune = []
        running_rewards = {k: 0 for k in np.unique(self.storage['keys'])}
        running_prune = {k: 0 for k in np.unique(self.storage['keys'])}
        for keys, rewards, is_terminal in zip(self.storage['keys'][::-1], self.storage['rewards'][::-1], self.storage['is_terminals'][::-1]):
            for key, reward in zip(keys[::-1], rewards[::-1]):
                if is_terminal:
                    running_rewards[key] = 0  # Reset at terminal state
                    if prune is not None: running_prune[key] = 0
                running_rewards[key] = reward + gamma * running_rewards[key]
                ret.append(running_rewards[key])
                if prune is not None:
                    running_prune[key] += 1
                    ret_prune.append(running_prune[key] > prune)
        ret = ret[::-1]
        ret = torch.tensor(ret, dtype=torch.float32)
        if prune is not None:
            ret_prune = ret_prune[::-1]
            ret_prune = torch.tensor(ret_prune, dtype=torch.bool)

        # Need to normalize AFTER propagation
        for r, p in zip(ret, ret_prune):
            # Don't include pruned rewards in update
            if p: self.running_statistics.update(r)
        ret = (ret - self.running_statistics.mean()) / (torch.sqrt(self.running_statistics.variance() + 1e-8))

        if prune is not None:
            return ret, ret_prune
        return ret

    def clear(self, clear_persistent=False):
        "Clear memory"
        for k in self.storage: self.storage[k].clear()
        if clear_persistent:
            for k in self.persistent_storage: self.persistent_storage[k].clear()


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
        num_features_per_node,
        modal_sizes,
        output_dim,
        feature_embed_dim=32,
        embed_dim=64,
        num_heads=4,
        action_std_init=.6,
        activation=F.tanh,
        num_mlps=1,
    ):
        super().__init__()

        # Base information
        self.num_features_per_node = num_features_per_node
        self.modal_sizes = modal_sizes
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
        self.feature_embed = nn.ModuleList([ nn.Linear(self.modal_sizes[i], self.feature_embed_dim) for i in range(len(self.modal_sizes)) ])

        # Embedding
        solo_features_len = self.feature_embed_dim * len(self.modal_sizes) + self.num_features_per_node
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
        running_idx = self.num_features_per_node
        ret = [entities[..., :running_idx]]
        for ms, fe in zip(self.modal_sizes, self.feature_embed):
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
        self_embed = self.self_embed(self_entity).unsqueeze(-2)
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
            num_features_per_node,
            modal_sizes,
            output_dim,
            model=EntitySelfAttention,
            action_std_init=.6,
            action_std_decay=.05,
            action_std_min=.1,
            epochs=80,
            epsilon_clip=.2,
            memory_gamma=.95,
            memory_prune=100,
            actor_lr=3e-4,
            critic_lr=1e-3,
            lr_gamma=1,
            update_maxbatch=torch.inf,
            update_batch=torch.inf,
            update_minibatch=torch.inf,
            update_load_level='batch',
            update_cast_level='minibatch',
            rs_nset=1e5,
            device='cpu',
            **kwargs,
    ):
        super().__init__()

        # Variables
        self.action_std = action_std_init
        self.action_std_decay = action_std_decay
        self.action_std_min = action_std_min
        self.epochs = epochs
        self.epsilon_clip = epsilon_clip
        self.memory_gamma = memory_gamma
        self.memory_prune = memory_prune

        # Runtime management
        self.update_maxbatch = update_maxbatch
        self.update_batch = update_batch
        self.update_minibatch = update_minibatch
        self.update_load_level = update_load_level
        self.update_cast_level = update_cast_level
        self.device = device

        # New policy
        self.actor = model(num_features_per_node=num_features_per_node, modal_sizes=modal_sizes, output_dim=output_dim, action_std_init=action_std_init, **kwargs)
        self.critic = model(num_features_per_node=num_features_per_node, modal_sizes=modal_sizes, output_dim=1, **kwargs)

        # Old policy
        self.actor_old = model(num_features_per_node=num_features_per_node, modal_sizes=modal_sizes, output_dim=output_dim, action_std_init=action_std_init, **kwargs)
        self.critic_old = model(num_features_per_node=num_features_per_node, modal_sizes=modal_sizes, output_dim=1, **kwargs)

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr},
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Memory
        self.memory = AdvancedMemoryBuffer(sum(modal_sizes), rs_nset=rs_nset)

        # Copy current weights
        self.update_old_policy()

        # To device
        self.to(self.device)

    ### Base overloads
    def to(self, device):
        self.device = device
        return super().to(device)

    ### Utility functions
    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def decay_action_std(self):
        self.action_std = max(self.action_std - self.action_std_decay, self.action_std_min)
        self.actor.set_action_std(self.action_std)
        self.actor_old.set_action_std(self.action_std)

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

    def act_macro(self, state, *, keys=None, max_batch=None, max_nodes=None):
        # Act
        if max_batch is not None:
            # Compute `max_batch` at a time with randomized `max_nodes`
            initialized = False
            for start_idx in range(0, state.shape[0], max_batch):
                action_sub, action_log_sub, state_val_sub = self.act(
                    *utilities.split_state(
                        state,
                        idx=list( range(start_idx, min(start_idx+max_batch, state.shape[0])) ),
                        max_nodes=max_nodes,
                    ),
                    return_all=True,
                )
                # TODO (Major): Record the split for use during update

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
            action, action_log, state_val = self.act(*utilities.split_state(state, max_nodes=max_nodes), return_all=True)

        # Record
        # NOTE: `reward` and `is_terminal` are added outside of the class, calculated
        # after stepping the environment
        if keys is not None and self.training:
            self.memory.record(
                keys=keys,
                states=state,  # Will cast to cpu after
                actions=action.cpu(),
                action_logs=action_log.cpu(),
                state_vals=state_val.cpu(),
            )

        return action

    def forward(self, *state):
        # Calculate action
        actions = self.actor.calculate_actions(state)
        action, _ = self.actor.select_action(actions)

        return action

    ### Backward functions
    def update(self):
        # Calculate rewards
        rewards = self.memory.propagate_rewards(gamma=self.memory_gamma, prune=self.memory_prune)
        if self.memory_prune is not None: rewards, rewards_mask = rewards
        else: rewards_mask = torch.ones(len(rewards), dtype=bool)
        rewards, rewards_mask = rewards.detach(), rewards_mask.detach()

        # Batch parameters
        level_dict = {'maxbatch': 0, 'batch': 1, 'minibatch': 2}
        load_level = level_dict[self.update_load_level]  # 0, 1, 2 : max, batch, mini
        cast_level = level_dict[self.update_cast_level]  # 0, 1, 2 : max, batch, mini
        assert cast_level >= load_level, 'Cannot cast without first loading'

        # Determine batch sizes
        memory_size = sum(rewards_mask)
        maxbatch_size = self.update_maxbatch if self.update_maxbatch is not None else memory_size
        maxbatch_size = int(min(maxbatch_size, memory_size))
        batch_size = self.update_batch if self.update_batch is not None else maxbatch_size
        batch_size = int(min(batch_size, maxbatch_size))
        minibatch_size = self.update_minibatch if self.update_minibatch is not None else batch_size
        minibatch_size = int(min(minibatch_size, batch_size))

        # Load maxbatch
        maxbatch_idx = np.random.choice(
            np.arange(len(self.memory))[rewards_mask],  # Only consider states which have rewards with significant future samples
            maxbatch_size,
            replace=False,
        )
        if load_level == 0:
            maxbatch_data = self.memory[maxbatch_idx]
            maxbatch_rewards = rewards[maxbatch_idx]
        if cast_level == 0:
            maxbatch_data = utilities.dict_map_recursive_tensor_idx_to(maxbatch_data, None, self.device)
            maxbatch_rewards = maxbatch_rewards.to(self.device)

        # Train
        for _ in range(self.epochs):
            # Load batch
            batch_idx = np.random.choice(maxbatch_size, batch_size, replace=False)
            batch_absolute_idx = maxbatch_idx[batch_idx]
            if load_level == 1:
                batch_data = self.memory[batch_absolute_idx]
                batch_rewards = rewards[batch_absolute_idx]
            elif load_level < 1:
                batch_data = utilities.dict_map_recursive_tensor_idx_to(maxbatch_data, batch_idx, None)
                batch_rewards = maxbatch_rewards[batch_idx]
            if cast_level == 1:
                batch_data = utilities.dict_map_recursive_tensor_idx_to(batch_data, None, self.device)
                batch_rewards = batch_rewards.to(self.device)

            # Gradient accumulation
            for _, min_idx in enumerate(range(0, batch_size, minibatch_size)):
                # Load minibatch
                max_idx = min(min_idx + minibatch_size, batch_size)
                minibatch_idx = np.arange(min_idx, max_idx)
                minibatch_absolute_idx = batch_absolute_idx[minibatch_idx]
                if load_level == 2:
                    minibatch_data = self.memory[minibatch_absolute_idx]
                    minibatch_rewards = rewards[minibatch_absolute_idx]
                if load_level < 2:
                    minibatch_data = utilities.dict_map_recursive_tensor_idx_to(batch_data, minibatch_idx, None)
                    minibatch_rewards = batch_rewards[minibatch_idx]
                if cast_level == 2:
                    minibatch_data = utilities.dict_map_recursive_tensor_idx_to(minibatch_data, None, self.device)
                    minibatch_rewards = minibatch_rewards.to(self.device)

                # Get subset data
                states_old_sub = minibatch_data['states']
                actions_old_sub = minibatch_data['actions']
                action_logs_old_sub = minibatch_data['action_logs']
                state_vals_old_sub = minibatch_data['state_vals']

                # Get subset rewards
                advantages_sub = minibatch_rewards - state_vals_old_sub

                # Evaluate actions
                action_logs, dist_entropy = self.actor.evaluate_action(states_old_sub, actions_old_sub)

                # Evaluate states
                state_vals = self.critic.evaluate_state(states_old_sub)

                # Ratio between new and old probabilities
                ratios = torch.exp(action_logs - action_logs_old_sub)

                # Calculate PPO loss
                unclipped = ratios * advantages_sub
                clipped = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages_sub
                loss_PPO = -torch.min(unclipped, clipped)

                # Calculate critic loss
                loss_critic = .5 * F.mse_loss(state_vals, minibatch_rewards)

                # Calculate entropy loss
                # TODO (Minor): Figure out purpose
                loss_entropy = -.01 * dist_entropy

                # CLI
                # print(f'Epoch {epoch+1:02} - Minibatch {minibatch+1:01}')
                # print(f'PPO: {loss_PPO.mean():.3f}, critic: {loss_critic.mean():.3f}, entropy: {loss_entropy.mean():.3f}')
                # print()

                # Calculate total loss
                loss = loss_PPO + loss_critic + loss_entropy
                loss = loss.mean()

                # Scale and calculate gradient
                accumulation_frac = (max_idx - min_idx) / batch_size
                loss = loss * accumulation_frac
                loss.backward()  # Longest computation

            # Step
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update scheduler
        self.scheduler.step()

        # Copy current weights
        self.update_old_policy()

        # Clear memory
        self.memory.clear()
