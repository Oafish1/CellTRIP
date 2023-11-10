import torch
import torch.nn as nn


class EntitySelfAttention(nn.Module):
    def __init__(self, num_features_per_node, output_dim, embed_dim=64, num_heads=4):
        super().__init__()

        self.num_features_per_node = num_features_per_node
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embedding
        self.self_embed = nn.Linear(self.num_features_per_node, self.embed_dim)
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Linear(self.embed_dim + self.num_features_per_node, self.embed_dim)

        # Self attention
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        # Decision
        self.decider = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, self_entity, node_entities):
        # Embed all entities
        self_embed = self.self_embed(self_entity)
        node_embeds = self.node_embed(torch.concat((self_embed.repeat(node_entities.shape[0], 1), node_entities), dim=1))
        embeds = torch.concat((self_embed, node_embeds), dim=0)

        # Self attention across entities
        attentions = self.self_attention(embeds, embeds, embeds)[0]
        embeddings = attentions.sum(dim=0)

        # Decision
        actions = torch.clip(self.decider(embeddings), -1, 1)

        return actions


class PPO(nn.Module):
    def __init__(self, num_features_per_node, output_dim, model=EntitySelfAttention, **kwargs):
        super().__init__()

        # New policy
        self.actor = model(num_features_per_node, output_dim, **kwargs)
        self.critic = model(num_features_per_node, 1, **kwargs)

        # Old policy
        self.actor_old = model(num_features_per_node, output_dim, **kwargs)
        self.critic_old = model(num_features_per_node, 1, **kwargs)


        self.update_old_policy()

    def update_old_policy(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def forward(self, *args):
        actions = self.actor(*args)
        actions_old = self.actor_old(*args)
        state_val = self.critic(*args)
        state_val_old = self.critic_old(*args)

        return actions, actions_old, state_val, state_val_old
