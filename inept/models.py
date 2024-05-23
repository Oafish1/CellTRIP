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
    def __init__(self, prefix_len):
        # User parameters
        self.prefix_len = prefix_len

        # Storage variables
        self.storage = {
            'keys': [],             # Lists containing keys in the first dim of states
            'states': [],           # State tensors of dim `keys x non-prefix features`
            'actions': [],          # Actions
            'action_logs': [],      # Action probabilities
            'state_vals': [],       # Critic evaluation of state
            'rewards': [],          # Rewards, list of lists
            'is_terminals': [],     # Booleans indicating if the terminal state has been reached
        }
        self.persistent_storage = {
            'prefixes': {},         # Prefixes corresponding to keys
            'prefix_matrices': {},  # Orderings of prefixes
        }

        # Cache
        self.prefix_matrix = {}

        # Maintenance variables
        self.recorded = {k: False for k in self.storage}

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
                        # `_append_prefix` takes most time without caching, then `split_state`
                        val = utilities.split_state(
                            self._append_prefix(self.storage[k][list_num], keys=self.storage['keys'][list_num]),
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

        # Sort to indexing order
        for k in ret:
            ret[k] = [ret[k][i] for i in sort_inverse_idx]

        return dict(ret)

    def __len__(self):
        return sum(len(keys) for keys in self.storage['keys'])

    def _append_prefix(self, state, *, keys, cache=True):
        "Append prefixes to state vector"
        # Read from cache
        if cache and str(keys) in self.persistent_storage['prefix_matrices']:
            prefix_matrix = self.persistent_storage['prefix_matrices'][str(keys)]

        else:
            # Aggregate prefixes
            prefix_matrix = None
            for k in keys:
                val = self.persistent_storage['prefixes'][k].unsqueeze(0)
                if prefix_matrix is None: prefix_matrix = val
                else: prefix_matrix = torch.concat((prefix_matrix, val), dim=0)

            # Add to cache
            if cache: self.persistent_storage['prefix_matrices'][str(keys)] = prefix_matrix

        # Append to state
        return torch.concat((prefix_matrix, state), dim=1)

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
            # Record
            self.storage[k].append(v)
            self.recorded[k] = True

        # Reset if all variables have been recorded
        if np.array([v for _, v in self.recorded.items()]).all():
            # Gleam prefixes
            for j, k in enumerate(self.storage['keys'][-1]):
                if k not in self.persistent_storage['prefixes']:
                    self.persistent_storage['prefixes'][k] = self.storage['states'][-1][j][:self.prefix_len]

            # Cut prefixes
            self.storage['states'][-1] = self.storage['states'][-1][:, self.prefix_len:]

            # Set all variables as unrecorded
            for k in self.recorded: self.recorded[k] = False

    def propagate_rewards(self, gamma=.99):
        "Propagate rewards with decay"
        ret = []
        running_rewards = {k: 0 for k in np.unique(self.storage['keys'])}
        for keys, rewards, is_terminal in zip(self.storage['keys'][::-1], self.storage['rewards'][::-1], self.storage['is_terminals'][::-1]):
            for key, reward in zip(keys[::-1], rewards[::-1]):
                if is_terminal: running_rewards[key] = 0  # Reset at terminal state
                running_rewards[key] = reward + gamma * running_rewards[key]
                ret.append(running_rewards[key])
        ret = ret[::-1]
        return torch.tensor(ret, dtype=torch.float32)

    def clear(self, clear_persistent=False):
        "Clear memory"
        for k in self.storage: self.storage[k].clear()
        if clear_persistent:
            for k in self.persistent_storage: self.persistent_storage[k].clear()


class MemoryBuffer:
    """
    Naïve implementation of memory
    """
    def __init__(self):
        self.keys = []  # Entity key
        self.states = []
        self.actions = []
        self.action_logs = []
        self.state_vals = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.keys)

    def propagate_rewards(self, gamma=.99):
        # Propagate rewards backwards with decay
        rewards = []
        running_rewards = {k: 0 for k in np.unique(self.keys)}
        for key, reward, is_terminal in zip(self.keys[::-1], self.rewards[::-1], self.is_terminals[::-1]):
            if is_terminal: running_rewards[key] = 0  # Reset at terminal state
            running_rewards[key] = reward + gamma * running_rewards[key]
            rewards.append(running_rewards[key])
        rewards = rewards[::-1]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return rewards

    def clear(self):
        del self.keys[:]
        del self.states[:]
        del self.actions[:]
        del self.action_logs[:]
        del self.state_vals[:]
        del self.rewards[:]
        del self.is_terminals[:]


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
        modal_sizes,
        num_features_per_node,
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
        self.modal_sizes = modal_sizes
        self.num_features_per_node = num_features_per_node
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
        running_idx = 0
        ret = []
        for ms, fe in zip(self.modal_sizes, self.feature_embed):
            # Record embedded features
            val = fe(entities[..., running_idx:(running_idx + ms)])
            ret.append(val)

            # Increment start idx
            running_idx += ms

        else:
            # Add rest of state matrix
            ret.append(entities[..., running_idx:])

        # Construct full matrix
        ret = torch.concat(ret, dim=-1)
        self.activation(ret)
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
        # TODO: Should layer norm be added here and for feature embedding?
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
            modal_sizes,
            num_features_per_node,
            output_dim,
            model=EntitySelfAttention,
            action_std_init=.6,
            action_std_decay=.05,
            action_std_min=.1,
            epochs=80,
            epsilon_clip=.2,
            actor_lr=3e-4,
            critic_lr=1e-3,
            lr_gamma=1,
            update_max_batch=torch.inf,
            update_minibatch=4e4,
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

        # Runtime management
        self.update_max_batch = update_max_batch
        self.update_minibatch = update_minibatch
        self.device = device

        # New policy
        self.actor = model(modal_sizes, num_features_per_node, output_dim, action_std_init=action_std_init, **kwargs)
        self.critic = model(modal_sizes, num_features_per_node, 1, **kwargs)

        # Old policy
        self.actor_old = model(modal_sizes, num_features_per_node, output_dim, action_std_init=action_std_init, **kwargs)
        self.critic_old = model(modal_sizes, num_features_per_node, 1, **kwargs)

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr},
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Memory
        self.memory = AdvancedMemoryBuffer(sum(modal_sizes))

        # Copy current weights
        self.update_old_policy()

        # To device
        self.to(self.device)

    ### Base overloads
    def to(self, device):
        super().to(device)
        self.device = device

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

    def act_macro(self, state, *, keys):
        # Act
        action, action_log, state_val = self.act(*utilities.split_state(state), return_all=True)

        # Record
        # NOTE: `reward` and `is_terminal` are added outside of the class, calculated
        # after stepping the environment
        self.memory.record(
            keys=keys,
            states=state.cpu(),
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
        # Propagate and normalize rewards
        rewards = self.memory.propagate_rewards().detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Train
        for _ in range(self.epochs):
            # Monte carlo sampling
            batch_idx = np.random.choice(len(self.memory), min(self.update_max_batch, len(self.memory)), replace=False)

            # Gradient accumulation
            # TODO: maybe spread out minibatches?
            for min_idx in range(0, len(batch_idx), self.update_minibatch):
                # Get minibatch idx
                max_idx = min(min_idx + self.update_minibatch, len(batch_idx))
                minibatch_idx = batch_idx[min_idx:max_idx]

                # Get subset data
                # 'states', 'actions', 'action_logs', 'state_vals'
                data = self.memory[minibatch_idx]  # This takes a long time, maybe a couple seconds
                states_old_sub = [torch.concat([s[i] for s in data['states']], dim=0).to(self.device) for i in range(2)]
                actions_old_sub = torch.stack(data['actions'], dim=0).to(self.device)
                action_logs_old_sub = torch.stack(data['action_logs'], dim=0).to(self.device)
                state_vals_old_sub = torch.stack(data['state_vals'], dim=0).to(self.device)

                # Get subset rewards
                rewards_sub = rewards[minibatch_idx].to(self.device)
                advantages_sub = rewards_sub - state_vals_old_sub

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
                loss_critic = .5 * F.mse_loss(state_vals, rewards_sub)

                # Calculate entropy loss
                # TODO: Figure out purpose
                loss_entropy = -.01 * dist_entropy

                # Calculate total loss
                loss = loss_PPO + loss_critic + loss_entropy
                loss = loss.mean()

                # Scale and calculate gradient
                accumulation_frac = (max_idx - min_idx) / len(batch_idx)
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
