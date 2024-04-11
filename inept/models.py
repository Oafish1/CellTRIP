import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F


### Utility classes
class MemoryBuffer:
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
        num_features_per_node,
        output_dim,
        embed_dim=64,
        num_heads=4,
        action_std_init=.6,
        activation=F.tanh,
        num_mlps=1,
    ):
        super().__init__()

        # Base information
        self.num_features_per_node = num_features_per_node
        self.output_dim = output_dim
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

        # Embedding
        self.self_embed = nn.Linear(self.num_features_per_node, self.embed_dim)
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Linear(self.embed_dim + self.num_features_per_node, self.embed_dim)

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
    def calculate_actions(self, state):
        # Formatting
        self_entity, node_entities = state
        # TODO: Perhaps node modalities should be encoded separately first

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
        self.actor = model(num_features_per_node, output_dim, action_std_init=action_std_init, **kwargs)
        self.critic = model(num_features_per_node, 1, **kwargs)

        # Old policy
        self.actor_old = model(num_features_per_node, output_dim, action_std_init=action_std_init, **kwargs)
        self.critic_old = model(num_features_per_node, 1, **kwargs)

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr},
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Memory
        self.memory = MemoryBuffer()

        # Copy current weights
        self.update_old_policy()

        # To device
        self.to(self.device)

    ### Utility functions
    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def decay_action_std(self):
        self.action_std = max(self.action_std - self.action_std_decay, self.action_std_min)
        self.actor.set_action_std(self.action_std)
        self.actor_old.set_action_std(self.action_std)

    ### Running functions
    def act(self, *state):
        # Add dimension if only one shape
        if len(state[0].shape) == 1:
            state = [s.unsqueeze[0] for s in state]

        # Calculate actions and state
        actions = self.actor.calculate_actions(state)
        action, action_log = self.actor.select_action(actions)
        state_val = self.critic.evaluate_state(state)

        # Record
        for key in range(state[0].shape[0]):
            self.memory.keys.append(key)
            self.memory.states.append([state[i][key].cpu() for i in range(len(state))])
            self.memory.actions.append(action[key].cpu())
            self.memory.action_logs.append(action_log[key].cpu())
            self.memory.state_vals.append(state_val[key].cpu())
        # NOTE: `reward` and `is_terminal` are added outside of the class, calculated
        # after stepping the environment

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

        # Format inputs
        states_old = [torch.stack([state[i] for state in self.memory.states], dim=0).detach() for i in range(len(self.memory.states[0]))]
        actions_old = torch.stack(self.memory.actions, dim=0).detach()
        action_logs_old = torch.stack(self.memory.action_logs, dim=0).detach()
        state_vals_old = torch.stack(self.memory.state_vals, dim=0).detach()

        # Calculate advantages
        advantages = rewards - state_vals_old

        # Send to GPU
        states_old = [states_old[i].to(self.device) for i in range(len(self.memory.states[0]))]  # remove .to(self.device) if too much
        rewards = rewards.to(self.device)
        advantages = advantages.to(self.device)
        actions_old = actions_old.to(self.device)
        action_logs_old = action_logs_old.to(self.device)
        # state_vals_old = state_vals_old.to(self.device)

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

                # Sample minibatches
                states_old_sub = [states_old[i][minibatch_idx] for i in range(len(self.memory.states[0]))]  # .to(self.device)
                rewards_sub = rewards[minibatch_idx]
                advantages_sub = advantages[minibatch_idx]
                actions_old_sub = actions_old[minibatch_idx]
                action_logs_old_sub = action_logs_old[minibatch_idx]

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
