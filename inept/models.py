import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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


### Policy classes
class EntitySelfAttention(nn.Module):
    def __init__(self, num_features_per_node, output_dim, embed_dim=64, num_heads=4, action_std_init=.6):
        super().__init__()

        # Base information
        self.num_features_per_node = num_features_per_node
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.set_action_std(action_std_init)

        # Embedding
        self.self_embed = nn.Linear(self.num_features_per_node, self.embed_dim)
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Linear(self.embed_dim + self.num_features_per_node, self.embed_dim)

        # Self attention
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        # Decision
        self.decider = nn.Linear(self.embed_dim, self.output_dim)

    ### Training functions
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = torch.full((self.output_dim,), new_action_std**2)

    ### Calculation functions
    def calculate_actions(self, state, selfish=True):
        # Formatting
        self_entity, node_entities = state
        # TODO: Perhaps node modalities should be encoded separately first

        # Debug
        if selfish:
            self_embed = self.self_embed(self_entity)
            self_embed = F.tanh(self_embed)
            actions = self.decider(self_embed)
            actions = F.tanh(actions)
            return actions

        # Embed all entities
        self_embed = self.self_embed(self_entity).unsqueeze(-2)
        self_embed = F.tanh(self_embed)
        node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_entities.shape[:-1], self_embed.shape[-1]), node_entities), dim=-1))
        node_embeds = F.tanh(node_embeds)
        embeds = torch.concat((self_embed, node_embeds), dim=-2)

        # Self attention across entities
        attentions = self.self_attention(embeds, embeds, embeds)[0]
        embeddings = attentions.sum(dim=1)  # Sum across entities
        embeddings = F.tanh(embeddings)

        # Decision
        actions = self.decider(embeddings)
        actions = F.tanh(actions)

        return actions

    def evaluate_state(self, state):
        return self.calculate_actions(state).squeeze(-1)

    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        covariance = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(actions, covariance)

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
            **kwargs,
    ):
        super().__init__()

        # Variables
        self.action_std = action_std_init
        self.action_std_decay = action_std_decay
        self.action_std_min = action_std_min
        self.epochs = epochs
        self.epsilon_clip = epsilon_clip

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

        # Memory
        self.memory = MemoryBuffer()

        # Copy current weights
        self.update_old_policy()

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
            self.memory.states.append([state[i][key] for i in range(len(state))])
            self.memory.actions.append(action[key])
            self.memory.action_logs.append(action_log[key])
            self.memory.state_vals.append(state_val[key])
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
        rewards = self.memory.propagate_rewards()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Format inputs
        states_old = [torch.stack([state[i] for state in self.memory.states], dim=0).detach() for i in range(len(self.memory.states[0]))]
        actions_old = torch.stack(self.memory.actions, dim=0).detach()
        action_logs_old = torch.stack(self.memory.action_logs, dim=0).detach()
        state_vals_old = torch.stack(self.memory.state_vals, dim=0).detach()

        # Calculate advantages
        advantages = rewards - state_vals_old

        # Train
        for _ in range(self.epochs):
            # Evaluate actions
            action_logs, dist_entropy = self.actor.evaluate_action(states_old, actions_old)

            # Evaluate states
            state_vals = self.critic.evaluate_state(states_old)

            # Ratio between new and old probabilities
            ratios = torch.exp(action_logs - action_logs_old)

            # Calculate PPO loss
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss_PPO = -torch.min(unclipped, clipped)

            # Calculate critic loss
            loss_critic = .5 * F.mse_loss(state_vals, rewards)

            # Calculate entropy loss
            # TODO: Figure out purpose
            loss_entropy = -.01 * dist_entropy

            # Calculate total loss
            loss = loss_PPO + loss_critic + loss_entropy
            loss = loss.mean()

            # Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy current weights
        self.update_old_policy()

        # Clear memory
        self.memory.clear()
