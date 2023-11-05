import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features

        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.decider = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, X, mask):
        # mask: True if is included
        attentions = self.self_attention(X, X, X, key_padding_mask=~mask)
        average_embeddings = attentions[mask].sum(axis=0)


class Node(nn.Module):
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
        average_embeddings = attentions.sum(dim=0)

        # Decision
