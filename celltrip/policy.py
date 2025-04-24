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


# Modules
class ResidualAttention(nn.Module):
    def __init__(
        self,
        num_dims,
        num_heads,
        activation=nn.ReLU(),
        num_mlps=1,
        **kwargs
    ):
        super().__init__()
        # Layers
        self.attention = nn.MultiheadAttention(num_dims, num_heads, batch_first=True, **kwargs)
        self.layer_norms = nn.ModuleList([ nn.LayerNorm(num_dims) for _ in range(2+num_mlps) ])

        # MLP
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_dims, num_dims),
                activation,
                nn.Linear(num_dims, num_dims))
            for _ in range(num_mlps)])

    def forward(self, x, kv=None, mask=None):
        # Parameters
        x1 = x
        layer_norm_idx = 0

        # Apply residual self attention
        x2 = self.layer_norms[layer_norm_idx](x1)
        # x2 = x1
        layer_norm_idx += 1
        if kv is None: kv = x2
        x3, _ = self.attention(x2, kv, kv, attn_mask=mask)
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
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.):
    torch.nn.init.sparse_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

class EntitySelfAttention(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        log_std_init=-1,
        feature_embed_dim=32,
        embed_dim=256,
        num_heads=4,
        activation=nn.ReLU(),
        num_mlps=1,
        independent_critic=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Base information
        self.positional_dim = positional_dim
        self.modal_dims = modal_dims
        self.output_dim = output_dim
        self.independent_critic = independent_critic
        # self.activation = activation

        # Log std
        self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float))

        # Feature embedding
        self.feature_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(modal_dims[i], feature_embed_dim),
                activation,
                nn.Linear(feature_embed_dim, feature_embed_dim),
                nn.LayerNorm(feature_embed_dim))
            for i in range(len(modal_dims))])
        # Embedding
        solo_features_len = feature_embed_dim * len(modal_dims) + positional_dim
        self.self_embed = nn.Sequential(
            nn.Linear(solo_features_len, embed_dim),
            activation,
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Sequential(
            nn.Linear(embed_dim + solo_features_len, embed_dim),
            activation,
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim))  # Not across entities
        # Self attention
        self.residual_self_attention = ResidualAttention(embed_dim, num_heads, activation=activation, num_mlps=num_mlps)
        # Critic
        if self.independent_critic:
                    # Feature embedding
            self.critic_feature_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(modal_dims[i], feature_embed_dim),
                    activation,
                    nn.Linear(feature_embed_dim, feature_embed_dim),
                    nn.LayerNorm(feature_embed_dim))
                for i in range(len(modal_dims))])
            # Embedding
            solo_features_len = feature_embed_dim * len(modal_dims) + positional_dim
            self.critic_self_embed = nn.Sequential(
                nn.Linear(solo_features_len, embed_dim),
                activation,
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))
            # This could be wrong, but I couldn't think of another interpretation.
            # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
            self.critic_node_embed = nn.Sequential(
                nn.Linear(embed_dim + solo_features_len, embed_dim),
                activation,
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))  # Not across entities
            # Self attention
            self.critic_residual_self_attention = ResidualAttention(embed_dim, num_heads, activation=activation, num_mlps=num_mlps)
        # Deciders
        self.actor_decider = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            activation,
            nn.Linear(embed_dim, output_dim))
        self.critic_decider = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            activation,
            nn.Linear(embed_dim, 1))

    def embed_features(self, entities, critic=False):
        # TODO: Maybe there's a more efficient way to do this with masking
        running_idx = self.positional_dim
        ret = [entities[..., :running_idx]]
        for ms, fe in zip(
            self.modal_dims,
            self.feature_embed if (not critic or not self.independent_critic) else self.critic_feature_embed):
            # Record embedded features
            val = fe(entities[..., running_idx:(running_idx + ms)])
            # val = self.activation(val)

            # Record
            ret.append(val)
            running_idx += ms # Increment start idx

        # Check shape
        assert running_idx == entities.shape[-1]

        # Construct full matrix
        ret = torch.concat(ret, dim=-1)

        return ret
    
    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        dist = MultivariateNormal(
            loc=actions,
            covariance_matrix=torch.diag(self.log_std.exp().square().expand((self.output_dim,))).unsqueeze(dim=0),
            # TODO: Double check no square here b/c Cholesky
            # scale_tril=torch.diag(self.log_std.exp().expand((self.output_dim,))).unsqueeze(dim=0),  # Speeds up computation compared to using cov matrix
            validate_args=False)  # Speeds up computation

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action)

        # Return
        ret = ()
        if not set_action:
            ret += (action,)
        ret += (action_log,)
        if return_entropy: ret += (dist.entropy(),)
        return ret
    
    def forward(
        self, self_entity, node_entities,
        actor=True, sample=False, action=None, entropy=False, critic=False):
        # Common block
        if actor or not self.independent_critic:
            # Feature embedding (could be done with less redundancy, also only debatably needed)
            self_embed = self.embed_features(self_entity)
            node_embeds = self.embed_features(node_entities)
            # Self embedding
            self_embed = self.self_embed(self_embed).unsqueeze(-2)  # This has blown up with multi-gpu backward
            # Node embeddings
            node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_embeds.shape[:-1], self_embed.shape[-1]), node_embeds), dim=-1))
            # Self attention across entities
            embeddings = torch.concat((self_embed, node_embeds), dim=-2)
            attentions = self.residual_self_attention(embeddings)
            attentions_pool = attentions.mean(dim=-2)  # Average across entities
            embedding = torch.concat((self_embed.squeeze(-2), attentions_pool), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)
            common_embedding = embedding
            # common_embedding = self_entity[:, :6]


        # Critic block
        if self.independent_critic and critic:
            # Feature embedding (could be done with less redundancy, also only debatably needed)
            self_embed = self.embed_features(self_entity, critic=True)
            node_embeds = self.embed_features(node_entities, critic=True)
            # Self embedding
            self_embed = self.critic_self_embed(self_embed).unsqueeze(-2)  # This has blown up with multi-gpu backward
            # Node embeddings
            node_embeds = self.critic_node_embed(torch.concat((self_embed.expand(*node_embeds.shape[:-1], self_embed.shape[-1]), node_embeds), dim=-1))
            # Self attention across entities
            embeddings = torch.concat((self_embed, node_embeds), dim=-2)
            attentions = self.critic_residual_self_attention(embeddings)
            attentions_pool = attentions.mean(dim=-2)  # Average across entities
            embedding = torch.concat((self_embed.squeeze(-2), attentions_pool), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)
            critic_embedding = embedding
            # critic_embedding = self_entity[:, :6]
        else: critic_embedding = common_embedding

        # Decisions, samples, and returns
        ret = ()
        if actor:
            actions = self.actor_decider(common_embedding)  # actions
            ret += (actions,)
            if sample:
                ret += self.select_action(actions, action=action, return_entropy=entropy)  # action, action_log, dist_entropy
        if critic: ret += (self.critic_decider(critic_embedding).squeeze(-1),)  # state_val
        return ret
    

class Dummy(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        log_std_init=-1,
        hidden_dim=128,
        embed_dim=32,
        activation=nn.ReLU,
        independent_critic=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Parameters
        self.positional_dim = positional_dim
        self.output_dim = output_dim
        self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float))
        self.independent_critic = independent_critic

        # Layers
        num_dims = positional_dim + np.array(modal_dims).sum()
        self.self_embed = nn.Sequential(
            nn.Linear(num_dims, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation())
        # Independent critic
        if independent_critic:
            self.critic_self_embed = nn.Sequential(
                nn.Linear(num_dims, hidden_dim), activation(),
                nn.Linear(hidden_dim, hidden_dim), activation())
        # Deciders
        self.actions = nn.Parameter(torch.eye(output_dim), requires_grad=False)
        self.action_embed = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, embed_dim), activation())
        self.actor_decider = nn.Sequential(
            activation(), nn.Linear(hidden_dim, embed_dim))
        self.critic_decider = nn.Sequential(
            activation(), nn.Linear(hidden_dim, 1))
    
    def forward(
            self, self_entities, node_entities=None, mask=None,
            actor=True, sample=False, action=None, entropy=False, critic=False, squeeze=True):
        # Formatting
        # if self_entities.dim() > 2:
        #     shape = self_entities.shape
        #     self_entities = self_entities.reshape((-1, self_entities.shape[-1]))
        # else: shape = None

        # Common block
        if actor or not self.independent_critic:
            self_embeds = self.self_embed(self_entities)  # Self embedding
            common_self_embeds = self_embeds

        # Critic block
        if self.independent_critic and critic:
            self_embeds = self.self_embed(self_entities)  # Self embedding
            critic_self_embeds = self_embeds
        else: critic_self_embeds = common_self_embeds

        # Decisions, samples, and returns
        ret = ()
        if actor:
            self_action_embeds = self.actor_decider(common_self_embeds)
            action_embeds = self.action_embed(self.actions)
            if self_action_embeds.dim() > 2:
                action_means = torch.einsum('bik,jk->bij', self_action_embeds, action_embeds)
            else: action_means = torch.einsum('ik,jk->ij', self_action_embeds, action_embeds)
            ret += (action_means,)
            if sample: ret += self.select_action(action_means, action=action, return_entropy=entropy)  # action, action_log, dist_entropy
        if critic: ret += (self.critic_decider(critic_self_embeds).squeeze(-1),)  # state_val
        # if shape is not None: ret = tuple(t.reshape(*shape[:2], -1) if t.dim() == 2 else t.reshape(*shape[:2]) for t in ret)
        if squeeze and self_entities.dim() > 2:
            ret = tuple(t.squeeze(dim=1) for t in ret)
            # ret = tuple(t.flatten(end_dim=1) for t in ret)
        
        return ret
    
    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        dist = MultivariateNormal(
            loc=actions,
            # covariance_matrix=torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0),
            # TODO: Double check no square here b/c Cholesky
            scale_tril=torch.diag(self.log_std.exp().expand((self.output_dim,))).unsqueeze(dim=0),  # Speeds up computation compared to using cov matrix
            validate_args=False)  # Speeds up computation

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action)

        # Return
        ret = ()
        if not set_action:
            ret += (action,)
        ret += (action_log,)
        if return_entropy: ret += (dist.entropy(),)
        return ret
    

def orthogonal_init(module):
    for name, param in module.named_parameters():
        if name.endswith('.weight') or name.endswith('_weight'):
            std = .1 if name.startswith('actor_decider') else 1
            try:
                torch.nn.init.orthogonal_(param, std)
                # print(f'Weight: {name} - {std}')
            except:
                torch.nn.init.constant_(param, 1)
                # print(f'Weight Fail: {name}')
        elif name.endswith('.bias') or name.endswith('_bias'):
            # print(f'Bias: {name}')
            torch.nn.init.constant_(param, 0)
        else:
            # print(f'None: {name}')
            pass


class ResidualAttentionBlock(nn.Module):
    def __init__(self, num_dims, num_heads, activation=nn.ReLU):
        super().__init__()
        # Layers
        self.attention = nn.MultiheadAttention(num_dims, num_heads, batch_first=True)
        self.norms = nn.ModuleList([ nn.LayerNorm(num_dims) for _ in range(3) ])

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_dims, 4*num_dims), activation(),
            nn.Linear(4*num_dims, num_dims))

    def forward(self, x, kv=None, mask=None):
        # Parameters
        if kv is None: x = kv

        # Residual Attention
        x1 = self.norms[0](x)
        # kv1 = self.norms[1](kv)
        kv1 = kv
        x2, _ = self.attention(x1, kv1, kv1, attn_mask=mask)
        x = x + x2
        x1 = self.norms[2](x)
        x2 = self.mlp(x1)
        x = x + x2
        return x


class EntitySelfAttentionLite(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        log_std_init=-1,
        hidden_dim=128,  # 256
        embed_dim=16,  # 32
        heads=4,  # 8
        blocks=2,  # 4
        activation=nn.ReLU,
        independent_critic=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Parameters
        self.positional_dim = positional_dim
        self.output_dim = output_dim
        self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float))
        self.heads = heads
        self.independent_critic = independent_critic

        # Actor layers
        num_feat_dims = np.array(modal_dims).sum()
        self.self_pos_embed = nn.Linear(positional_dim, hidden_dim)
        self.self_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
        self.node_pos_embed = nn.Linear(positional_dim, hidden_dim)
        self.node_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
        self.self_embed = nn.Sequential(
            activation(), nn.Linear(hidden_dim, hidden_dim), activation())
        self.node_embed = nn.Sequential(
            activation(), nn.Linear(hidden_dim, hidden_dim), activation())
        self.residual_attention_blocks = nn.ModuleList([
            ResidualAttentionBlock(hidden_dim, heads, activation=activation) for _ in range(blocks)])
        
        # Independent critic layers
        if independent_critic:
            self.critic_self_pos_embed = nn.Linear(positional_dim, hidden_dim)
            self.critic_self_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
            self.critic_node_pos_embed = nn.Linear(positional_dim, hidden_dim)
            self.critic_node_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
            self.critic_self_embed = nn.Sequential(
                activation(), nn.Linear(hidden_dim, hidden_dim), activation())
            self.critic_node_embed = nn.Sequential(
                activation(), nn.Linear(hidden_dim, hidden_dim), activation())
            self.critic_residual_attention_blocks = nn.ModuleList([
                ResidualAttentionBlock(hidden_dim, heads, activation=activation) for _ in range(blocks)])
        # Deciders
        self.actions = nn.Parameter(torch.eye(output_dim), requires_grad=False)
        self.action_embed = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, embed_dim), activation())
        self.actor_decider = nn.Sequential(
            activation(), nn.Linear(hidden_dim, embed_dim))
        self.critic_decider = nn.Sequential(
            activation(), nn.Linear(hidden_dim, 1))
    
    def forward(
            self, self_entities, node_entities=None, mask=None,
            actor=True, sample=False, action=None, entropy=False, critic=False,
            feature_embeds=None, return_feature_embeds=False,
            squeeze=True, fit_and_strip=True):
        # Formatting
        if node_entities is None: node_entities = self_entities
        if mask is None:
            mask = torch.eye(self_entities.shape[-2], dtype=torch.bool, device=self_entities.device)
            if self_entities.dim() > 2: mask = mask.repeat((self_entities.shape[0], 1, 1))
        if self_entities.dim() > 2:
            # NOTE: Grouped batches (i.e. (>1, >1, ...) shape) are possible with squeeze=False
            if mask.dim() < 3: mask.unsqueeze(0)
            # mask = mask.repeat((self.heads, 1, 1))
            mask = mask[[i for i in range(mask.shape[0]) for _ in range(self.heads)]]
        if feature_embeds is not None:
            feature_embeds_ret = feature_embeds
            feature_embeds = feature_embeds.copy()
        else: feature_embeds_ret = []

        # Actor block
        if actor or not self.independent_critic:
            # Feature embedding
            self_pos_embeds = self.self_pos_embed(self_entities[..., :self.positional_dim])
            node_pos_embeds = self.node_pos_embed(node_entities[..., :self.positional_dim])
            if feature_embeds is not None: self_feat_embeds, node_feat_embeds = feature_embeds.pop(0)
            else:
                self_feat_embeds = self.self_feat_embed(self_entities[..., self.positional_dim:])
                node_feat_embeds = self.node_feat_embed(node_entities[..., self.positional_dim:])
                feature_embeds_ret.append((self_feat_embeds, node_feat_embeds))
            # Self embeddings
            self_embeds = self.self_embed(self_pos_embeds+self_feat_embeds)
            # Node embeddings
            node_embeds = self.node_embed(node_pos_embeds+node_feat_embeds)
            # Attention
            for block in self.residual_attention_blocks:
                self_embeds = block(self_embeds, kv=node_embeds, mask=mask)
            actor_self_embeds = self_embeds

        # Critic block
        if self.independent_critic and critic:
            # Feature embedding
            self_pos_embeds = self.critic_self_pos_embed(self_entities[..., :self.positional_dim])
            node_pos_embeds = self.critic_node_pos_embed(node_entities[..., :self.positional_dim])
            if feature_embeds is not None: self_feat_embeds, node_feat_embeds = feature_embeds.pop(0)
            else:
                self_feat_embeds = self.critic_self_feat_embed(self_entities[..., self.positional_dim:])
                node_feat_embeds = self.critic_node_feat_embed(node_entities[..., self.positional_dim:])
                feature_embeds_ret.append((self_feat_embeds, node_feat_embeds))
            # Self embeddings
            self_embeds = self.critic_self_embed(self_pos_embeds+self_feat_embeds)
            # Node embeddings
            node_embeds = self.critic_node_embed(node_pos_embeds+node_feat_embeds)
            # Attention
            for block in self.critic_residual_attention_blocks:
                self_embeds = block(self_embeds, kv=node_embeds, mask=mask)
            actor_self_embeds = self_embeds
        else: critic_self_embeds = actor_self_embeds

        # NOTE: fit_and_strip breaks compatibility on batch-batch native programs (Grouped batches)
        if fit_and_strip and self_entities.dim() > 2:
            # Strip padded entries
            actor_self_embeds = actor_self_embeds[~actor_self_embeds.sum(dim=-1).isnan()]
            if critic: critic_self_embeds = critic_self_embeds[~critic_self_embeds.sum(dim=-1).isnan()]

        # Decisions, samples, and returns
        ret = ()
        if actor:  # action_means
            self_action_embeds = self.actor_decider(actor_self_embeds)
            action_embeds = self.action_embed(self.actions)
            # Norms
            self_action_embeds = self_action_embeds / self_action_embeds.norm(p=2, keepdim=True, dim=-1)
            action_embeds = action_embeds / action_embeds.norm(p=2, keepdim=True, dim=-1)
            # Dot/cosine
            if self_action_embeds.dim() > 2:
                action_means = torch.einsum('bik,jk->bij', self_action_embeds, action_embeds)
            else: action_means = torch.einsum('ik,jk->ij', self_action_embeds, action_embeds)
            ret += (action_means,)
            if sample: ret += self.select_action(action_means, action=action, return_entropy=entropy)  # action, action_log, dist_entropy
        if critic: ret += (self.critic_decider(critic_self_embeds).squeeze(-1),)  # state_vals
        if squeeze and self_entities.dim() > 2 and not fit_and_strip:
            ret = tuple(t.flatten(0, 1) for t in ret)
        if return_feature_embeds: ret += feature_embeds_ret,
        return ret
    
    def select_action(self, actions, *, action=None, return_entropy=False):
        # Format
        set_action = action is not None

        # Select continuous action
        if self.training: scale_tril = torch.diag(self.log_std.exp().expand((self.output_dim,))).unsqueeze(dim=0)
        else: scale_tril = torch.zeros((self.output_dim, self.output_dim), device=self.log_std.device)
        dist = MultivariateNormal(
            loc=actions,
            # covariance_matrix=torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0),
            # TODO: Double check no square here b/c Cholesky
            scale_tril=scale_tril,  # Speeds up computation compared to using cov matrix
            validate_args=False)  # Speeds up computation

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action)
        if return_entropy: entropy = dist.entropy()

        # Return
        ret = ()
        if not set_action: ret += action,
        ret += action_log,
        if return_entropy: ret += entropy,
        return ret

        # # Feature embedding
        # self.feature_embed = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(modal_dims[i], feature_embed_dim),
        #         activation)
        #     for i in range(len(modal_dims))])

        # # Embedding
        # solo_features_len = feature_embed_dim * len(modal_dims) + positional_dim
        # self.self_embed = nn.Sequential(
        #     nn.Linear(solo_features_len, embed_dim),
        #     activation,
        #     nn.Linear(embed_dim, embed_dim))
        # # This could be wrong, but I couldn't think of another interpretation.
        # # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        # self.node_embed = nn.Sequential(
        #     nn.Linear(solo_features_len, embed_dim),
        #     activation,
        #     nn.Linear(embed_dim, embed_dim))  # Not across entities

        # # Self attention
        # self.residual_attention = ResidualAttention(embed_dim, num_heads, activation=activation, num_mlps=num_mlps)

        # # Deciders
        # self.actor_decider = nn.Sequential(
        #     layer_init(nn.Linear(embed_dim, embed_dim)),
        #     activation,
        #     layer_init(nn.Linear(embed_dim, output_dim), std=1e-2))
        # self.critic_decider = nn.Sequential(
        #     layer_init(nn.Linear(embed_dim, embed_dim)),
        #     activation,
        #     layer_init(nn.Linear(embed_dim, 1), std=1.))

    # def embed_features(self, entities):
    #     running_idx = self.positional_dim
    #     ret = [entities[..., :running_idx]]
    #     for ms, fe in zip(
    #         self.modal_dims,
    #         self.feature_embed):
    #         # Record embedded features
    #         val = fe(entities[..., running_idx:(running_idx + ms)])
    #         # val = self.activation(val)

    #         # Record
    #         ret.append(val)
    #         running_idx += ms # Increment start idx

    #     # Check shape
    #     assert running_idx == entities.shape[-1]

    #     # Construct full matrix
    #     ret = torch.concat(ret, dim=-1)

    #     return ret


# class EntityCalculation(nn.Module):
#     def __init__(self, output_dim, log_std_init=-1, **kwargs):
#         super().__init__()

#         # Parameters
#         self.output_dim = output_dim

#         # Set action std
#         self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float))
#         # self.recalculate_scale_tril()

#     # def recalculate_scale_tril(self):
#     #     covariance_matrix = torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0)
#     #     self.scale_tril = torch.linalg.cholesky(covariance_matrix)
#     #     print(self.scale_tril)

#     def calculate_actions(self, state): pass

#     def evaluate_state(self, state):
#         return self.calculate_actions(state).squeeze(-1)

#     def evaluate_action(self, state, action):
#         actions = self.calculate_actions(state)
#         action_log, dist_entropy = self.select_action(actions, action=action, return_entropy=True)

#         return action_log, dist_entropy

#     def select_action(self, actions, *, action=None, return_entropy=False):
#         # Format
#         set_action = action is not None

#         # Select continuous action
#         dist = MultivariateNormal(
#             loc=actions,
#             # covariance_matrix=torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0),
#             # TODO: Double check no square here b/c Cholesky
#             scale_tril=torch.diag(self.log_std.exp().expand((self.output_dim,))).unsqueeze(dim=0),  # Speeds up computation compared to using cov matrix
#             validate_args=False)  # Speeds up computation

#         # Sample
#         if not set_action: action = dist.sample()
#         action_log = dist.log_prob(action)

#         # Return
#         ret = ()
#         if not set_action:
#             ret += (action,)
#         ret += (action_log,)
#         if return_entropy: ret += (dist.entropy(),)
#         if len(ret) == 1: ret = ret[0]
#         return ret


# class EntitySelfAttention(EntityCalculation):
#     def __init__(
#         self,
#         positional_dim,
#         modal_dims,
#         output_dim,
#         feature_embed_dim=32,
#         embed_dim=64,
#         num_heads=4,
#         activation=nn.ReLU(),
#         final_activation=nn.Tanh(),
#         num_mlps=1,
#         critic=False,
#         **kwargs,
#     ):
#         super().__init__(output_dim, **kwargs)

#         # Base information
#         self.positional_dim = positional_dim
#         self.modal_dims = modal_dims
#         self.output_dim = output_dim
#         self.feature_embed_dim = feature_embed_dim
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.activation = activation
#         self.final_activation = final_activation
#         self.num_mlps = num_mlps
#         self.critic = critic

#         # Feature embedding
#         self.feature_embed = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.modal_dims[i], self.feature_embed_dim),
#                 self.activation,
#                 nn.Linear(self.feature_embed_dim, self.feature_embed_dim),
#                 nn.LayerNorm(self.feature_embed_dim))
#             for i in range(len(self.modal_dims))])

#         # Embedding
#         solo_features_len = self.feature_embed_dim * len(self.modal_dims) + self.positional_dim
#         self.self_embed = nn.Sequential(
#             nn.Linear(solo_features_len, self.embed_dim),
#             self.activation,
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim))
#         # This could be wrong, but I couldn't think of another interpretation.
#         # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
#         self.node_embed = nn.Sequential(
#             nn.Linear(self.embed_dim + solo_features_len, self.embed_dim),
#             self.activation,
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim))  # Not across entities

#         # Self attention
#         self.residual_self_attention = ResidualAttention(self.embed_dim, self.num_heads, activation=self.activation, num_mlps=self.num_mlps)

#         # Decision
#         self.decider = nn.Sequential(
#             nn.Linear(2*self.embed_dim, 2*self.embed_dim),
#             self.activation,
#             nn.Linear(2*self.embed_dim, self.output_dim))

#     ### Calculation functions
#     def _embed_features(self, entities):
#         # TODO: Maybe there's a more efficient way to do this with masking
#         running_idx = self.positional_dim
#         ret = [entities[..., :running_idx]]
#         for ms, fe in zip(
#             self.modal_dims,
#             self.feature_embed):
#             # Record embedded features
#             # Maybe pre-ln if we don't care about magnitude
#             val = fe(entities[..., running_idx:(running_idx + ms)])
#             val = self.activation(val)

#             # Record
#             ret.append(val)
#             running_idx += ms # Increment start idx

#         # Check shape
#         assert running_idx == entities.shape[-1]

#         # Construct full matrix
#         ret = torch.concat(ret, dim=-1)

#         return ret
    
#     def calculate_actions(self, state):
#         # Formatting
#         self_entity, node_entities = state

#         # Feature embedding (could be done with less redundancy)
#         self_entity = self._embed_features(self_entity)
#         node_entities = self._embed_features(node_entities)

#         # Self embedding
#         self_embed = self.self_embed(self_entity).unsqueeze(-2)  # This has blown up with multi-gpu backward

#         # Node embeddings
#         node_embeds = self.node_embed(torch.concat((self_embed.expand(*node_entities.shape[:-1], self_embed.shape[-1]), node_entities), dim=-1))

#         # Self attention across entities
#         embeddings = torch.concat((self_embed, node_embeds), dim=-2)
#         attentions = self.residual_self_attention(embeddings)
#         attentions_pool = attentions.mean(dim=-2)  # Average across entities
#         embedding = torch.concat((self_embed.squeeze(-2), attentions_pool), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)

#         # Decision
#         # TODO (Minor): Should layer norm be added here and for feature embedding? Probably not.
#         actions = self.decider(embedding)
#         if self.final_activation is not None: actions = self.final_activation(actions)

#         return actions
    

# class EntitySelfAttentionLite(EntityCalculation):
#     def __init__(
#         self,
#         positional_dim,
#         modal_dims,
#         output_dim,
#         feature_embed_dim=32,
#         embed_dim=64,
#         num_heads=4,
#         activation=nn.ReLU(),
#         final_activation=nn.Tanh(),
#         num_blocks=1,
#         num_mlps=1,
#         critic=False,
#         **kwargs,
#     ):
#         super().__init__(output_dim, **kwargs)

#         # Base information
#         self.positional_dim = positional_dim
#         self.modal_dims = modal_dims
#         self.output_dim = output_dim
#         self.feature_embed_dim = feature_embed_dim
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.activation = activation
#         self.final_activation = final_activation
#         self.num_mlps = num_mlps
#         self.critic = critic

#         # Feature embedding
#         self.feature_embed = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.modal_dims[i], self.feature_embed_dim),
#                 self.activation,
#                 nn.Linear(self.feature_embed_dim, self.feature_embed_dim),
#                 nn.LayerNorm(self.feature_embed_dim))
#             for i in range(len(self.modal_dims))])

#         # Embedding
#         solo_features_len = self.feature_embed_dim * len(self.modal_dims) + self.positional_dim
#         self.q_embed = nn.Sequential(
#             nn.Linear(solo_features_len, self.embed_dim),
#             self.activation,
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim))
#         self.kv_embed = nn.Sequential(
#             nn.Linear(solo_features_len, self.embed_dim),
#             self.activation,
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.LayerNorm(self.embed_dim))

#         # Self attention
#         self.residual_attention_blocks = nn.ModuleList([
#             ResidualAttention(self.embed_dim, self.num_heads, activation=self.activation, num_mlps=self.num_mlps) for _ in range(num_blocks)])

#         # Decision
#         self.decider = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             self.activation,
#             nn.Linear(self.embed_dim, self.output_dim))

#     ### Calculation functions
#     def embed_features(self, entities):
#         running_idx = self.positional_dim
#         ret = [entities[..., :running_idx]]
#         for ms, fe in zip(
#             self.modal_dims,
#             self.feature_embed):
#             # Record embedded features
#             # Maybe pre-ln if we don't care about magnitude
#             val = fe(entities[..., running_idx:(running_idx + ms)])
#             val = self.activation(val)

#             # Record
#             ret.append(val)
#             running_idx += ms # Increment start idx

#         # Check shape
#         assert running_idx == entities.shape[-1]

#         # Construct full matrix
#         ret = torch.concat(ret, dim=-1)

#         return ret
    
#     def calculate_actions(self, state):
#         # return torch.zeros((200, 3), device=self.action_std.device)

#         # Formatting
#         entities, = state

#         # Feature embedding (could be done with less redundancy)
#         entities = self.embed_features(entities)

#         # Self embedding
#         query = self.q_embed(entities).unsqueeze(0)
#         key_value = self.kv_embed(entities).unsqueeze(0)

#         # Self attention across entities
#         for block in self.residual_attention_blocks:
#             query = block(query, key_value)
#         # attentions_pool = attentions.mean(dim=-2)  # Average across entities
#         # embedding = torch.concat((self_embed.squeeze(0), attentions.squeeze(0)), dim=-1)  # Concatenate self embedding to pooled embedding (pg. 24)

#         # Decision
#         # TODO (Minor): Should layer norm be added here and for feature embedding? Probably not.
#         actions = self.decider(query.squeeze(0))
#         if self.final_activation is not None: actions = self.final_activation(actions)

#         return actions


class MomentumStandardization(nn.Module):
    def __init__(self, momentum=0, mean=0, std=1):  # .99997
        super().__init__()

        self.momentum = momentum
        self.mean = nn.Parameter(torch.tensor(float(mean)), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(float(std)), requires_grad=False)

    def update(self, x):
        "First dimension must be batch"
        samples = x.shape[0]
        self.mean.data = self.momentum**samples * self.mean + (1 - self.momentum**samples) * x.mean(keepdim=True, dim=0)
        self.std.data = self.momentum**samples * self.std + (1 - self.momentum**samples) * x.std(keepdim=True, dim=0)  # Not technically correct

    def apply(self, x, mean=True, std=True):
        if mean: x = x - self.mean
        if std: x = x / self.std
        return x

    def remove(self, x, mean=True, std=True):
        if std: x = x * self.std
        if mean: x = x + self.mean
        return x


### Training classes
class PPO(nn.Module):
    def __init__(
            self,
            positional_dim,
            modal_dims,
            output_dim,
            # Model Parameters
            # model=EntitySelfAttention, # sample_strategy 'random-proximity'
            model=EntitySelfAttentionLite,  # sample_strategy None
            # Forward
            log_std_init=0,
            forward_batch_size=int(5e4),
            vision_size=int(1e2),
            # sample_strategy='random-proximity',
            sample_strategy=None,
            sample_dim=None,
            reproducible_strategy='mean',
            # Weights
            epsilon_ppo=.2,
            epsilon_critic=torch.inf,  # Formerly .1
            critic_weight=.5,
            entropy_weight=1e-2,
            kl_beta_init=0.,
            kl_beta_increment=(.5, 2),
            kl_target=.03,
            kl_early_stop=False,
            grad_clip=torch.inf,
            # Optimizers
            log_std_lr=3e-4,
            actor_lr=3e-4,
            critic_lr=3e-5,  # Not really needed
            weight_decay=0.,
            betas=(.9, .999),  # (.997, .997),
            # lr_iters=200,
            lr_gamma=.97,
            # Backward
            update_iterations=10,  # 10*int(32*4*200/(64*200)), MIGHT WANT TO CHANGE, CHANGED FROM 80
            sync_iterations=1,
            pool_size=None,
            epoch_size=8*2_000,
            batch_size=2_000,
            minibatch_size=None,
            load_level='minibatch',
            cast_level='minibatch',
            **kwargs,
    ):
        super().__init__()

        # Parameters
        self.positional_dim = positional_dim
        self.modal_dims = modal_dims.copy()
        self.output_dim = output_dim

        # Weights
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

        # Variables
        self.epsilon_ppo = epsilon_ppo
        self.epsilon_critic = epsilon_critic
        self.kl_beta = nn.Parameter(torch.tensor(kl_beta_init), requires_grad=False)
        self.kl_beta_increment = kl_beta_increment
        self.kl_target = kl_target
        self.kl_early_stop = kl_early_stop
        self.grad_clip = grad_clip

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

        # New policy
        self.actor_critic = model(positional_dim, modal_dims, output_dim, log_std_init=log_std_init, **kwargs)
        self.log_std_params = list(filter(lambda kv: kv[0] in ('log_std',), self.actor_critic.named_parameters()))
        self.critic_params = list(filter(lambda kv: kv[0].startswith('critic_'), self.actor_critic.named_parameters()))
        claimed_params = list(map(lambda kv: kv[0], self.log_std_params)) + list(map(lambda kv: kv[0], self.critic_params))
        self.actor_params = list(filter(lambda kv: kv[0] not in claimed_params, self.actor_critic.named_parameters()))
        # self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.log_std_params, 'lr': log_std_lr},
            {'params': self.actor_params, 'lr': actor_lr},
            {'params': self.critic_params, 'lr': critic_lr}],
            betas=betas, weight_decay=weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=lr_iters)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Old policy
        self.actor_critic_old = model(positional_dim, modal_dims, output_dim, log_std_init=log_std_init, **kwargs)
        log_std_params = list(filter(lambda kv: kv[0] in ('log_std',), self.actor_critic_old.named_parameters()))
        critic_params = list(filter(lambda kv: kv[0].startswith('critic_'), self.actor_critic_old.named_parameters()))
        claimed_params = list(map(lambda kv: kv[0], log_std_params)) + list(map(lambda kv: kv[0], critic_params))
        actor_params = list(filter(lambda kv: kv[0] not in claimed_params, self.actor_critic_old.named_parameters()))
        # self.optimizer_old = torch.optim.Adam(self.actor_critic_old.parameters(), lr=actor_lr)
        self.optimizer_old = torch.optim.Adam([
            {'params': log_std_params, 'lr': log_std_lr},
            {'params': actor_params, 'lr': actor_lr},
            {'params': critic_params, 'lr': critic_lr}],
            betas=betas, weight_decay=weight_decay)

        # Non-grad params
        self.policy_iteration = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.return_standardization = MomentumStandardization()

        # Initialize
        orthogonal_init(self)
        self.copy_policy()

    def initialize(self):
        for name, param in self.named_parameters():
            if name.endswith('.weight') or name.endswith('_weight'):
                std = .1 if name.startswith('actor_decider') else 1
                try:
                    nn.init.orthogonal_(param, std)
                    # print(f'Weight: {name} - {std}')
                except:
                    nn.init.constant_(param, 1)
                    # print(f'Weight Fail: {name}')
            elif name.endswith('.bias') or name.endswith('_bias'):
                # print(f'Bias: {name}')
                nn.init.constant_(param, 0)
            else:
                # print(f'None: {name}')
                pass

    # def to(self, device):
    #     ret = super().to(device)
    #     self.device = self.policy_iteration.device
    #     return ret

    def get_log_std(self):
        return self.actor_critic.log_std.detach().item()

    def get_advantage_std(self):
        return self.return_standardization.std.mean().item()

    def get_policy_iteration(self):
        return int(self.policy_iteration.item())
    
    def copy_policy(self, _invert=False):
        "Copy new policy weights onto old policy"
        sources, targets = (
            (self.actor_critic, self.optimizer),
            (self.actor_critic_old, self.optimizer_old))
        if _invert: sources, targets = targets, sources
        for source, target in zip(sources, targets):
            target.load_state_dict(source.state_dict())

    def revert_policy(self):
        "Copy old policy weights onto new policy"
        self.copy_policy(_invert=True)

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
            with s3.open(fname, 'rb') as f:
                policy_state = torch.load(f, map_location=self.policy_iteration.device)
        else:
            policy_state = torch.load(fname, map_location=self.policy_iteration.device)

        # Load policy
        _utility.general.set_policy_state(self, policy_state)

    def forward(
        self, compressed_state, *,
        keys=None, memory=None, forward_batch_size=None, terminal=False,
        feature_embeds=None, return_feature_embeds=False):
        # Data Checks
        assert compressed_state.shape[0] > 0, 'Empty state matrix passed'
        if keys is not None: assert len(keys) == compressed_state.shape[0], (
            f'Length of keys vector must equal state dimension 0 ({compressed_state.shape[0]}), '
            f'got {len(keys)} instead.'
        )
            
        # Defaults
        if forward_batch_size is None: forward_batch_size = self.forward_batch_size
        feature_embeds_arg = feature_embeds
        construct_feature_embeds = feature_embeds is None and return_feature_embeds

        # Act
        action = torch.zeros(0, device=self.policy_iteration.device)
        action_log = torch.zeros(0, device=self.policy_iteration.device)
        state_val = torch.zeros(0, device=self.policy_iteration.device)
        for start_idx in range(0, compressed_state.shape[0], forward_batch_size):
            state = _utility.processing.split_state(
                compressed_state,
                idx=np.arange(start_idx, min(start_idx+forward_batch_size, compressed_state.shape[0])),
                **self.split_args)
            if not terminal:
                _, action_sub, action_log_sub, state_val_sub, feature_embeds_sub = self.actor_critic(
                    *state, sample=True, critic=True, feature_embeds=feature_embeds_arg, return_feature_embeds=True)
                action = torch.concat((action, action_sub), dim=0)
                action_log = torch.concat((action_log, action_log_sub), dim=0)
            else: state_val_sub, feature_embeds_sub = self.actor_critic(
                *state, actor=False, critic=True, feature_embeds=feature_embeds_arg, return_feature_embeds=True)
            state_val = torch.concat((state_val, state_val_sub), dim=0)
            if construct_feature_embeds:
                if feature_embeds is None: feature_embeds = feature_embeds_sub
                else:
                    feature_embeds = [
                        tuple(torch.concat((feature_embeds[i][j], t)) for j, t in enumerate(feat_tensors))
                        for i, feat_tensors in enumerate(feature_embeds_sub)]

        # Unstandardize state_val
        state_val = self.return_standardization.remove(state_val, mean=False)
        
        # Record
        # NOTE: `reward` and `is_terminal` are added outside of the class, calculated
        # after stepping the environment
        if memory is not None and keys is not None:  #  and self.training
            if not terminal:
                memory.record_buffer(
                    keys=keys,
                    states=compressed_state,
                    actions=action,
                    action_logs=action_log,
                    state_vals=state_val)
            else:
                memory.record_buffer(
                    terminal_states=compressed_state,
                    terminal_state_vals=state_val)
        ret = action,
        if return_feature_embeds: ret += feature_embeds,
        return _utility.general.clean_return(ret)
    
    def frozen_update(self, *args, **kwargs):
        # Init
        ret = ()

        # Actor update
        changed_params = [param for _, param in self.critic_params if param.requires_grad]
        for param in changed_params: param.requires_grad = False
        ret += self.update(*args, **kwargs)
        for param in changed_params: param.requires_grad = True

        # Critic update
        changed_params = [param for _, param in self.actor_params if param.requires_grad]
        changed_params += [param for _, param in self.log_std_params if param.requires_grad]
        for param in changed_params: param.requires_grad = False
        ret += self.update(*args, **kwargs)
        for param in changed_params: param.requires_grad = True
        return ret

        # for name, param in self.named_parameters():
        #     print(f'{name}: {param.requires_grad}')

    def update(
        self,
        memory,
        update_iterations=None,
        standardize_returns=False,
        verbose=False,
        # Collective args
        sync_iterations=None,
        **kwargs,
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
        pool_size = self.pool_size
        if pool_size is not None: pool_size = int(min(pool_size, memory_size))
        epoch_size = self.epoch_size if not (self.epoch_size is None and pool_size is not None) else pool_size
        if epoch_size is not None and pool_size is not None: epoch_size = int(min(epoch_size, pool_size))
        batch_size = self.batch_size if not (self.batch_size is None and epoch_size is not None) else epoch_size
        if batch_size is not None and epoch_size is not None: batch_size = int(min(batch_size, epoch_size))
        minibatch_size = self.minibatch_size if not (self.minibatch_size is None and batch_size is not None) else batch_size
        if minibatch_size is not None and batch_size is not None: minibatch_size = int(min(minibatch_size, batch_size))
        if sync_iterations == 1:
            # Adjust batch size if gradients are synchronized
            if batch_size is not None: batch_size = np.ceil(batch_size / self.get_world_size('learners')).astype(int)
            else: minibatch_size = np.ceil(minibatch_size / self.get_world_size('learners')).astype(int)

        # Cap at max size to reduce redundancy for sequential samples
        # NOTE: Pool->epoch is the only non-sequential sample, and is thus not included here
        # max_unique_memories = batch_size * update_iterations
        # epoch_size = min(epoch_size, max_unique_memories)

        # Update moving return mean
        if standardize_returns:
            # TODO: Clean this up if it works
            advantages = torch.concat([memory.storage['advantages'][i] for i in range(memory.get_steps()) if memory.storage['staleness'][i]==0]).to(self.policy_iteration.device)
            state_vals = torch.concat([memory.storage['state_vals'][i] for i in range(memory.get_steps()) if memory.storage['staleness'][i]==0]).to(self.policy_iteration.device)
            self.return_standardization.update(advantages - state_vals)

        # Load pool
        total_losses = defaultdict(lambda: [])
        pool_data = _utility.processing.sample_and_cast(
            memory, None, None, pool_size,
            current_level=0, load_level=load_level, cast_level=cast_level,
            device=self.policy_iteration.device, **kwargs)

        # Train
        iterations = 0; synchronized = True; escape = False
        while True:
            # Load epoch
            epoch_data = _utility.processing.sample_and_cast(
                memory, pool_data, pool_size, epoch_size,
                current_level=1, load_level=load_level, cast_level=cast_level,
                device=self.policy_iteration.device, **kwargs)
            batches = np.ceil(epoch_size/batch_size).astype(int) if epoch_size is not None else 1
            for batch_num in range(batches):
                # Load batch
                batch_losses = defaultdict(lambda: 0)
                batch_data = _utility.processing.sample_and_cast(
                    memory, epoch_data, epoch_size, batch_size,
                    current_level=2, load_level=load_level, cast_level=cast_level,
                    device=self.policy_iteration.device, sequential_num=batch_num,
                    **kwargs)
                minibatches = np.ceil(batch_size/minibatch_size).astype(int) if batch_size is not None else 1
                for minibatch_num in range(minibatches):
                    # Load minibatch
                    minibatch_data = _utility.processing.sample_and_cast(
                        memory, batch_data, batch_size, minibatch_size,
                        current_level=3, load_level=load_level, cast_level=cast_level,
                        device=self.policy_iteration.device, sequential_num=minibatch_num,
                        **kwargs)

                    # Get subset data
                    states = minibatch_data['states']
                    actions = minibatch_data['actions']
                    action_logs = minibatch_data['action_logs']
                    state_vals = minibatch_data['state_vals']
                    advantages = minibatch_data['advantages']
                    # rewards = minibatch_data['propagated_rewards']

                    # Perform backward
                    loss, loss_ppo, loss_critic, loss_entropy, loss_kl = self.backward(
                        states, actions, action_logs, state_vals, advantages=advantages, rewards=None)

                    # Scale and calculate gradient
                    accumulation_frac = states[0].shape[0] / batch_size
                    loss = loss * accumulation_frac
                    loss.backward()  # Longest computation

                    # Scale and record
                    batch_losses['Total'] += loss.detach()
                    batch_losses['PPO'] += loss_ppo.detach().mean() * accumulation_frac
                    batch_losses['critic'] += loss_critic.detach().mean() * accumulation_frac
                    batch_losses['entropy'] += loss_entropy.detach().mean() * accumulation_frac
                    batch_losses['KL'] += loss_kl.detach().mean() * accumulation_frac
                
                # Record
                for k, v in batch_losses.items(): total_losses[k].append(v)

                # Synchronize GPU policies and step
                # NOTE: Synchronize gradients every batch if =1, else synchronize whole model
                # NOTE: =1 keeps optimizers in sync without need for whole-model synchronization
                if sync_iterations == 1: self.synchronize('learners', grad=True)  # Sync only grad
                if self.kl_early_stop and synchronized: self.copy_policy()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if sync_iterations != 1:
                    # Synchronize for offsets
                    sync_loop = (iterations) % sync_iterations == 0
                    last_epoch = iterations == update_iterations
                    if use_collective and (sync_loop or last_epoch):
                        self.synchronize('learners')
                        synchronized = True
                    else: synchronized = False

                # Update KL beta
                # NOTE: Same as Torch KLPENPPOLoss implementation
                if self.kl_early_stop or self.kl_beta != 0:
                    loss_kl_mean = loss_kl.detach().mean()
                    self.synchronize('learners', sync_list=[loss_kl_mean])
                    if not self.kl_early_stop:
                        exp_limit = 32
                        if loss_kl_mean < self.kl_target / 1.5 and self.kl_beta > 2**-exp_limit: self.kl_beta.data *= self.kl_beta_increment[0]
                        elif loss_kl_mean > self.kl_target * 1.5 and self.kl_beta < 2**exp_limit: self.kl_beta.data *= self.kl_beta_increment[1]
                # Escape and roll back if KLD too high
                if self.kl_early_stop:
                    if loss_kl_mean > 1.5 * self.kl_target:
                        if iterations - sync_iterations > 0:
                            # Revert to previous synchronized state within kl target
                            self.revert_policy()
                            # iterations -= sync_iterations
                            escape = True; break
                        else:
                            warnings.warn(
                                'Update exceeded KL target too fast! Proceeding with update, but may be unstable. '
                                'Try lowering clip or learning rate parameters.')
                            escape = True; break
                        
            # Iterate
            iterations += 1
            if iterations >= update_iterations: escape = True

            # CLI
            if verbose and (iterations in (1, 5) or iterations % 10 == 0 or escape):
                print(
                    f'Iteration {iterations:02} - '
                    + f' + '.join([f'{k} ({np.mean([v.item() for v in vl[-batches:]]):.5f})' for k, vl in total_losses.items()])
                    + f' :: Log STD ({self.get_log_std():.5f})'
                    + f' :: Advantage STD ({self.return_standardization.std.mean().item():.5f})')

            # Break
            if escape: break

        # Update scheduler
        self.scheduler.step()
        # Update records
        self.policy_iteration += 1
        self.copy_policy()
        # Return
        return iterations, {k: np.mean([v.item() for v in vl]) for k, vl in total_losses.items()}

    def get_world_size(self, group='default'):
        try:
            world_size = col.get_collective_group_size(group)
            if world_size == -1: raise RuntimeError
            return world_size
        except:
            warnings.warn(f'No group "{group}" found.')
            return 1

    def synchronize(self, group='default', sync_list=None, grad=False, broadcast=None, allreduce=None):
        # Defaults
        if broadcast is None: broadcast = False
        if allreduce is None: allreduce = not broadcast

        # Collective operations
        world_size = self.get_world_size(group)
        if world_size == 1: return

        # Sync
        sync_list = self.parameters() if sync_list is None else sync_list
        with torch.no_grad():
            for w in sync_list:  # zip(self.state_dict(), self.parameters())
                if grad: w = w.grad  # No in-place modification here
                if w is None: continue
                if w.dtype == torch.long: continue
                if broadcast: col.broadcast(w, 0, group)
                if allreduce:
                    col.allreduce(w, group)
                    if not grad: w /= world_size

    def backward(
        self,
        states,
        actions,
        action_logs,
        state_vals,
        advantages=None,
        rewards=None):
        if advantages is not None:
            # Get inferred rewards
            rewards = advantages + state_vals
            # print(f'{self.get_policy_iteration()} - {advantages_mean:.3f} - {advantages_std:.3f}')
        elif rewards is not None:
            # Get advantages
            advantages = rewards - state_vals
        # Get normalized advantages
        advantages_mean, advantages_std = advantages.mean(), advantages.std() + 1e-8
        normalized_advantages = (advantages - advantages_mean) / advantages_std

        # Get normalized returns/rewards (sensitive to minibatch)
        # self.return_standardization.update(rewards)
        # NOTE: Action STD explosion/instability points to a normalization and/or critic fitting issue
        #       or, possibly, the returns are too homogeneous - i.e. the problem is solved
        normalized_rewards = self.return_standardization.apply(rewards, mean=False)

        # Evaluate actions and states
        _, action_logs_new, dist_entropy, state_vals_new = self.actor_critic(*states, sample=True, action=actions, entropy=True, critic=True)
        # _, action_logs_new, state_vals_new = self.actor_critic(states, sample=True, action=actions, entropy=False, critic=True)
        # dist_entropy = -action_logs_new  # Approximation
        
        # Calculate PPO loss
        ratios = torch.exp(action_logs_new - action_logs)
        unclipped_ppo = ratios * normalized_advantages
        clipped_ppo = torch.clamp(ratios, 1-self.epsilon_ppo, 1+self.epsilon_ppo) * normalized_advantages
        loss_ppo = -torch.min(unclipped_ppo, clipped_ppo)

        # Calculate KL divergence
        # NOTE: A bit odd when it comes to replay
        # Discrete
        # loss_kl = F.kl_div(action_logs, action_logs_new, reduction='batchmean', log_target=True)
        # loss_kl = ((action_logs - action_logs_new)  # * action_logs.exp()).sum(-1)  # Approximation
        # Continuous (http://joschu.net/blog/kl-approx.html)
        log_ratio = (action_logs_new - action_logs)
        loss_kl = (log_ratio.exp() - 1) - log_ratio

        # Calculate critic loss
        # unclipped_critic = (state_vals_new - normalized_rewards).square()
        unclipped_critic = F.smooth_l1_loss(state_vals_new, normalized_rewards)
        clipped_state_vals_new = torch.clamp(state_vals_new, state_vals-self.epsilon_critic, state_vals+self.epsilon_critic)
        # clipped_critic = (clipped_state_vals_new - normalized_rewards).square()
        clipped_critic = F.smooth_l1_loss(clipped_state_vals_new, normalized_rewards, reduction='none')
        loss_critic = torch.max(unclipped_critic, clipped_critic)
        # if torch.rand(1) < .03:
        #     print(
        #         f'{state_vals_new.mean().detach().item():.3f}'
        #         f' - {rewards.mean().detach().item():.3f}'
        #         f' - {normalized_rewards.mean().detach().item():.3f}'
        #         f' - {normalized_rewards.std().detach().item():.3f}')

        # TODO: Calculate critic KL from Normal

        # Calculate entropy bonus
        # NOTE: Not included in training grad if action_std is constant
        loss_entropy = -dist_entropy

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

        return loss, loss_ppo, loss_critic, loss_entropy, loss_kl
