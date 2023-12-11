import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn


### Utility classes
class MemoryBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_logs = []
        self.state_vals = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.action_logs[:]
        del self.state_vals[:]
        del self.rewards[:]
        del self.is_terminals[:]


### Policy classes
class EntitySelfAttention(nn.Module):
    def __init__(self, num_features_per_node, output_dim, embed_dim=64, num_heads=4, action_std_init=1.):
        super().__init__()

        # Base information
        self.num_features_per_node = num_features_per_node
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.action_std_init = action_std_init

        # Action variance
        self.set_action_std(self.action_std_init)

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
        self.action_var = torch.full((self.output_dim,), new_action_std**2)

    ### Calculation functions
    def calculate_actions(self, state):
        # Formatting
        self_entity, node_entities = state
        # TODO: Perhaps node modalities should be encoded separately first

        # Embed all entities
        self_embed = self.self_embed(self_entity).unsqueeze(-2)
        node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_entities.shape[:-1], self_embed.shape[-1]), node_entities), dim=-1))
        embeds = torch.concat((self_embed, node_embeds), dim=-2)

        # Self attention across entities
        attentions = self.self_attention(embeds, embeds, embeds)[0]
        embeddings = attentions.sum(dim=0)

        # Decision
        actions = self.decider(embeddings)

        return actions

    def calculate_state(self, state):
        return self.calculate_actions(state).squeeze()

    def select_action(self, actions, *, action=None):
        # Format
        set_action = action is not None

        # Select continuous action
        covariance = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(actions, covariance)

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action)

        if not set_action:
            return action, action_log
        else:
            return action_log

    def evaluate_action(self, state, action):
        actions = self.calculate_actions(state)
        action_log = self.select_action(actions, action=action)

        return action_log


### Training classes
class PPO(nn.Module):
    def __init__(self, num_features_per_node, output_dim, model=EntitySelfAttention, **kwargs):
        super().__init__()

        # New policy
        self.actor = model(num_features_per_node, output_dim, **kwargs)
        self.critic = model(num_features_per_node, 1, **kwargs)

        # Old policy
        self.actor_old = model(num_features_per_node, output_dim, **kwargs)
        self.critic_old = model(num_features_per_node, 1, **kwargs)

        # Memory
        self.memory = MemoryBuffer()

        self.update_old_policy()

    ### Utility functions
    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    ### Running functions
    def act(self, *state):
        # Calculate actions and state
        actions = self.actor.calculate_actions(state)
        action, action_log = self.actor.select_action(actions)
        state_val = self.critic.calculate_state(state)

        # Record
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.action_logs.append(action_log)
        self.memory.state_vals.append(state_val)

        # TODO: Calculate `reward` and `is_terminal`, likely in another function
        # NOTE: Perhaps reward can just be calculated on current state, for calculation
        # time concerns

        return action

    def forward(self, *state):
        # Calculate action
        actions = self.actor.calculate_actions(state)
        action, _ = self.actor.select_action(actions)

        return action

    ### Backward functions
    def update(self):
        # TODO: Implement
        pass
