from collections import defaultdict
import io
import os
import warnings

import numpy as np
import ray.util.collective as col  # Maybe conditional import?
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utility as _utility


# Initialization
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.):
    torch.nn.init.sparse_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Modules
class Dummy(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        log_std_init=-1,
        hidden_dim=128,
        embed_dim=32,
        activation=nn.PReLU,
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
        dist = torch.distributions.MultivariateNormal(
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


class ResidualAttention(nn.Module):
    def __init__(
        self,
        num_dims,
        num_heads,
        activation=nn.PReLU,
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
                activation(),
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
        activation=nn.PReLU,
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
                activation(),
                nn.Linear(feature_embed_dim, feature_embed_dim),
                nn.LayerNorm(feature_embed_dim))
            for i in range(len(modal_dims))])
        # Embedding
        solo_features_len = feature_embed_dim * len(modal_dims) + positional_dim
        self.self_embed = nn.Sequential(
            nn.Linear(solo_features_len, embed_dim),
            activation(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        # This could be wrong, but I couldn't think of another interpretation.
        # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
        self.node_embed = nn.Sequential(
            nn.Linear(embed_dim + solo_features_len, embed_dim),
            activation(),
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
                    activation(),
                    nn.Linear(feature_embed_dim, feature_embed_dim),
                    nn.LayerNorm(feature_embed_dim))
                for i in range(len(modal_dims))])
            # Embedding
            solo_features_len = feature_embed_dim * len(modal_dims) + positional_dim
            self.critic_self_embed = nn.Sequential(
                nn.Linear(solo_features_len, embed_dim),
                activation(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))
            # This could be wrong, but I couldn't think of another interpretation.
            # Also backed by https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf
            self.critic_node_embed = nn.Sequential(
                nn.Linear(embed_dim + solo_features_len, embed_dim),
                activation(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim))  # Not across entities
            # Self attention
            self.critic_residual_self_attention = ResidualAttention(embed_dim, num_heads, activation=activation, num_mlps=num_mlps)
        # Deciders
        self.actor_decider = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            activation(),
            nn.Linear(embed_dim, output_dim))
        self.critic_decider = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            activation(),
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
        dist = torch.distributions.MultivariateNormal(
            loc=actions,
            # covariance_matrix=torch.diag(self.log_std.exp().square().expand((self.output_dim,))).unsqueeze(dim=0),
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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, num_dims, num_heads, activation=nn.PReLU):
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
        kv1 = self.norms[1](kv)
        # kv1 = kv
        x2, _ = self.attention(x1, kv1, kv1, attn_mask=mask)
        x = x + x2
        x1 = self.norms[2](x)
        x2 = self.mlp(x1)
        x = x + x2
        return x
    

class TransposeLastIf3D(nn.Module):
    def forward(self, X):
        if X.dim() > 2: return torch.transpose(X, -2, -1)
        return X


class EntitySelfAttentionLite(nn.Module):
    def __init__(
        self,
        positional_dim,
        modal_dims,
        output_dim,
        # Parameters
        log_std_init=0,
        # Deep
        # hidden_dim=256,
        # heads=8,
        # blocks=4,
        # Basic
        # hidden_dim=128,
        # heads=4,
        # blocks=2,
        # Barebones
        # hidden_dim=64,
        hidden_dim=None,
        heads=2,
        blocks=1,
        # Options
        activation=nn.PReLU,
        dropout=0.,
        # Structure
        discrete=False,
        # independent_critic=True,
        independent_critic=False,
        **kwargs,
    ):
        # TODO: Implement https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/cbp_linear.py#L83
        super().__init__(**kwargs)

        # Defaults
        hidden_dim = 2*positional_dim if hidden_dim is None else hidden_dim

        # Parameters
        self.positional_dim = positional_dim
        self.output_dim = output_dim
        self.heads = heads
        self.independent_critic = independent_critic

        # Actor layers
        num_feat_dims = np.array(modal_dims).sum()
        self.self_pos_embed = nn.Linear(positional_dim, hidden_dim)
        self.self_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
        # self.self_feat_embed = nn.Sequential(
        #     nn.Linear(num_feat_dims, 2*num_feat_dims),
        #     activation(), nn.Dropout(dropout), nn.Linear(2*num_feat_dims, hidden_dim))
        self.node_pos_embed = nn.Linear(positional_dim, hidden_dim)
        self.node_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
        # self.node_feat_embed = nn.Sequential(
        #     nn.Linear(num_feat_dims, 2*num_feat_dims),
        #     activation(), nn.Dropout(dropout), nn.Linear(2*num_feat_dims, hidden_dim))
        self.self_embed = nn.Sequential(
            activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.node_embed = nn.Sequential(
            activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.residual_attention_blocks = nn.ModuleList([
            ResidualAttentionBlock(hidden_dim, heads, activation=activation) for _ in range(blocks)])
        
        # Independent critic layers
        if independent_critic:
            self.critic_self_pos_embed = nn.Linear(positional_dim, hidden_dim)
            self.critic_self_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
            # self.critic_self_feat_embed = nn.Sequential(
            #     nn.Linear(num_feat_dims, 2*num_feat_dims),
            #     activation(), nn.Dropout(dropout), nn.Linear(2*num_feat_dims, hidden_dim))
            self.critic_node_pos_embed = nn.Linear(positional_dim, hidden_dim)
            self.critic_node_feat_embed = nn.Linear(num_feat_dims, hidden_dim)
            # self.critic_node_feat_embed = nn.Sequential(
            #     nn.Linear(num_feat_dims, 2*num_feat_dims),
            #     activation(), nn.Dropout(dropout), nn.Linear(2*num_feat_dims, hidden_dim))
            self.critic_self_embed = nn.Sequential(
                activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            self.critic_node_embed = nn.Sequential(
                activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            self.critic_residual_attention_blocks = nn.ModuleList([
                ResidualAttentionBlock(hidden_dim, heads, activation=activation) for _ in range(blocks)])
            
        # Dot method
        # self.actions = nn.Parameter(torch.eye(output_dim), requires_grad=False)
        # self.action_embed = nn.Sequential(
        #     nn.Linear(output_dim, embed_dim), activation())
        # self.actor_decider = nn.Sequential(
        #     activation(), nn.Linear(hidden_dim, embed_dim))
        
        # General method
        if discrete: self.actor_decider = DiscreteActions(hidden_dim, output_dim*(3,), activation=activation, dropout=dropout)
        else: self.actor_decider = ContinuousActions(hidden_dim, output_dim, activation=activation, dropout=dropout, log_std_init=log_std_init)
        
        # Magnitude method
        # self.actor_decider_direction = nn.Sequential(
        #     activation(), nn.Linear(hidden_dim, output_dim))  # , nn.Tanh()
        # self.actor_decider_magnitude = nn.Sequential(
        #     activation(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # Critic
        self.critic_decider = nn.Sequential(
            # activation(), nn.Linear(hidden_dim, hidden_dim),
            activation(), nn.Linear(hidden_dim, 1))
        
        # Input and output layers for standardization functions
        self.input_pos_layers = [self.self_pos_embed, self.node_pos_embed]
        if self.independent_critic: self.input_pos_layers += [self.critic_self_pos_embed, self.critic_node_pos_embed]
        self.input_feat_layers = [self.self_feat_embed, self.node_feat_embed]
        if self.independent_critic: self.input_feat_layers += [self.critic_self_feat_embed, self.critic_node_feat_embed]
        # self.input_feat_layers = [self.self_feat_embed[0], self.node_feat_embed[0]]
        # if self.independent_critic: self.input_feat_layers += [self.critic_self_feat_embed[0], self.critic_node_feat_embed[0]]
        self.output_critic_layers = [self.critic_decider[-1]]
    
    def forward(
            self, self_entities, node_entities=None, mask=None,
            actor=True, action=None, entropy=False, critic=False,
            feature_embeds=None, return_feature_embeds=False,
            squeeze=True, fit_and_strip=True):
        # Formatting
        if node_entities is None: node_entities = self_entities
        if mask is None:
            mask = torch.eye(self_entities.shape[-2], dtype=torch.bool, device=self_entities.device)
            if self_entities.dim() > 2: mask = mask.repeat((self_entities.shape[0], 1, 1))

        # Workaround for https://github.com/pytorch/pytorch/issues/41508
        # Essentially, totally masked entries in attn_mask will cause NAN on backprop
        # regardless of if they're even considered in loss, so, we allow masked entries
        # to attend to themselves
        include_mask = mask.sum(dim=-1) < mask.shape[-1]
        mask[~include_mask, torch.argwhere(~include_mask)[:, -1]] = False
        
        # More formatting
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
            # Positional embedding
            self_pos_embeds = self.self_pos_embed(self_entities[..., :self.positional_dim])
            node_pos_embeds = self.node_pos_embed(node_entities[..., :self.positional_dim])
            # Feature embedding
            if feature_embeds is not None: self_feat_embeds, node_feat_embeds = feature_embeds.pop(0)
            else:
                self_feat_embeds = self.self_feat_embed(self_entities[..., self.positional_dim:])
                node_feat_embeds = self.node_feat_embed(node_entities[..., self.positional_dim:])
                feature_embeds_ret.append((self_feat_embeds, node_feat_embeds))
            # Self embeddings
            self_embeds = self.self_embed(self_pos_embeds+self_feat_embeds)
            # self_embeds = self.self_embed(self_pos_embeds)
            # Node embeddings
            node_embeds = self.node_embed(node_pos_embeds+node_feat_embeds)
            # node_embeds = self.self_embed(node_pos_embeds)
            # Attention
            for block in self.residual_attention_blocks:
                self_embeds = block(self_embeds, kv=node_embeds, mask=mask)
            actor_self_embeds = self_embeds

        # Critic block
        if self.independent_critic and critic:
            # Positional embedding
            self_pos_embeds = self.critic_self_pos_embed(self_entities[..., :self.positional_dim])
            node_pos_embeds = self.critic_node_pos_embed(node_entities[..., :self.positional_dim])
            # Feature embedding
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
            critic_self_embeds = self_embeds
        else: critic_self_embeds = actor_self_embeds

        # NOTE: fit_and_strip breaks compatibility on batch-batch native programs (Grouped batches)
        if fit_and_strip and self_entities.dim() > 2:
            # Strip padded entries with workaround
            if actor: actor_self_embeds = actor_self_embeds[include_mask]
            if critic: critic_self_embeds = critic_self_embeds[include_mask]
            # # Strip padded entries
            # actor_self_embeds = actor_self_embeds[~actor_self_embeds.sum(dim=-1).isnan()]
            # if critic: critic_self_embeds = critic_self_embeds[~critic_self_embeds.sum(dim=-1).isnan()]

        # Decisions, samples, and returns
        ret = ()
        if actor:  # action_means
            # Dot Method
            # self_action_embeds = self.actor_decider(actor_self_embeds)
            # action_embeds = self.action_embed(self.actions)
            # # Norms
            # # NOTE: Norm causes NAN, could use eps but might as well remove to avoid vanish
            # self_action_embeds = self_action_embeds  / (self_action_embeds.norm(p=2, keepdim=True, dim=-1) + 1e-8)
            # action_embeds = action_embeds   / (action_embeds.norm(p=2, keepdim=True, dim=-1) + 1e-8)
            # # Dot/cosine
            # if self_action_embeds.dim() > 2:
            #     action_means = torch.einsum('bik,jk->bij', self_action_embeds, action_embeds)
            # else: action_means = torch.einsum('ik,jk->ij', self_action_embeds, action_embeds)

            # Regular method
            ret += self.actor_decider(actor_self_embeds, action=action, return_entropy=entropy)  # action, action_log, dist_entropy
            
            # Magnitude method
            # action_direction = self.actor_decider_direction(actor_self_embeds)
            # action_direction = action_direction / action_direction.norm(keepdim=True, dim=-1)  # Get direction unit
            # action_magnitude = self.actor_decider_magnitude(actor_self_embeds)
            # action_means = action_magnitude * action_direction
            # # action_direction = np.sqrt(action_direction.shape[-1]) * action_direction / action_direction.std(keepdim=True, dim=-1)  # Get into comparable range for sampling
            # # action_means = torch.concat((action_direction, action_magnitude), dim=-1)
        if critic: ret += (self.critic_decider(critic_self_embeds).squeeze(-1),)  # state_vals
        if squeeze and self_entities.dim() > 2 and not fit_and_strip:
            ret = tuple(t.flatten(0, 1) for t in ret)
        if return_feature_embeds: ret += feature_embeds_ret,
        return ret
    

class DiscreteActions(nn.Module):
    # Discretized based on advice from https://arxiv.org/pdf/2004.00980
    def __init__(self, input_dim, output_dims, hidden_dim=None, activation=nn.PReLU, dropout=0., **kwargs):
        super().__init__(**kwargs)

        # Params
        if hidden_dim is None: hidden_dim = input_dim

        # Heads
        self.deciders = nn.ModuleList([
            nn.Sequential(
                # # Normal
                # activation(), nn.Linear(input_dim, hidden_dim),
                # activation(), nn.Linear(hidden_dim, output_dim),

                # Layer norm + Dropout
                nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim),
                activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),

                # Batch norm + Dropout
                # nn.BatchNorm1d(input_dim), activation(), nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim), activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
            )
            for output_dim in output_dims])
        
    def forward(self, logits, *, action=None, return_entropy=False):
        # Calculate actions
        actions = torch.stack([decider(logits) for decider in self.deciders], dim=-2)

        # Format
        set_action = action is not None

        # Define normal distribution
        dist = torch.distributions.Categorical(logits=actions)  # /np.sqrt(actions.shape[-1])

        # Sample
        if not set_action:
            if self.training: action = dist.sample()
            else: action = actions.argmax(dim=-1)
        action_log = dist.log_prob(action).sum(dim=-1)  # Multiply independent probabilities
        if return_entropy: entropy = dist.entropy().sum(dim=-1)  # Technically, `sum` would be correct, but this is better for consistent weighting

        # Return
        ret = ()
        if not set_action: ret += action,
        ret += action_log,
        if return_entropy: ret += entropy,
        return ret


class ContinuousActions(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, activation=nn.PReLU, dropout=0., log_std_init=0, **kwargs):
        super().__init__(**kwargs)

        # Params
        self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float))
        if hidden_dim is None: hidden_dim = 2*input_dim
        
        # Heads
        self.decider = nn.Sequential(
            # Normal
            # activation(), nn.Linear(input_dim, hidden_dim),
            # activation(), nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),

            # Layer norm + Dropout
            activation(), nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim),
            activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),

            # Batch norm + Dropout
            # nn.BatchNorm1d(input_dim), activation(), nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(input_dim), activation(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),
        )
        
    def forward(self, logits, *, action=None, return_entropy=False):
        # Calculate actions
        actions = self.decider(logits)

        # Format
        set_action = action is not None

        # Select continuous action
        # if self.training: scale_tril = torch.diag(self.log_std.exp().expand((actions.shape[-1],)))  # .unsqueeze(dim=0)
        # else: scale_tril = torch.zeros((self.output_dim, self.output_dim), device=self.log_std.device)
        # scale_tril = torch.zeros((self.output_dim, self.output_dim), device=self.log_std.device)
        # dist = MultivariateNormal(
        #     loc=actions,
        #     # covariance_matrix=torch.diag(self.action_std.square().expand((self.output_dim,))).unsqueeze(dim=0),
        #     # TODO: Double check no square here b/c Cholesky
        #     scale_tril=scale_tril,  # Speeds up computation compared to using cov matrix
        #     validate_args=False)  # Speeds up computation

        # Define normal distribution
        std = self.log_std.exp() if self.training else 1e-7
        # std = self.log_std.exp()
        # std = 1e-7
        dist = torch.distributions.Normal(loc=actions, scale=std)  # /np.sqrt(actions.shape[-1])

        # Sample
        if not set_action: action = dist.sample()
        action_log = dist.log_prob(action).sum(dim=-1)  # Multiply independent probabilities
        if return_entropy: entropy = dist.entropy().sum(dim=-1)  # Technically, `sum` would be correct, but this is better for consistent weighting

        # Return
        ret = ()
        if not set_action: ret += action,
        ret += action_log,
        if return_entropy: ret += entropy,
        return ret


class ArtStandardization(nn.Module):
    "Adaptively Rescaling Targets"
    def __init__(self, dim=(1,), beta=3e-4, use_mean=True, use_std=True):
        super().__init__()

        self.beta = beta
        self.use_mean = use_mean
        self.use_std = use_std
        # TODO: Change to buffers
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)  # Should probably make this a buffer
        self.square_mean = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim).sqrt(), requires_grad=False)

    def update(self, x):
        "First dimension must be batch"
        # https://github.com/opendilab/PPOxFamily/blob/main/chapter4_reward/popart.py#L93
        # NOTE: It might be better to use another variance approximation method, but this one
        # is the easiest - https://www.johndcook.com/blog/2008/09/26/comparing-three-methods-of-computing-standard-deviation/
        # Update params
        batch_mean = (1-self.beta) * self.mean + self.beta * x.mean(dim=0)
        batch_square_mean = (1-self.beta) * self.square_mean + self.beta * x.square().mean(dim=0)
        batch_std = (batch_square_mean - batch_mean.square()).sqrt()

        # Handle NANs
        batch_mean[batch_mean.isnan()] = self.mean[batch_mean.isnan()]
        batch_square_mean[batch_square_mean.isnan()] = self.square_mean[batch_square_mean.isnan()]
        batch_std[batch_std.isnan()] = self.std[batch_std.isnan()]
        # batch_std = batch_std.nan_to_num(0.)
        batch_std = batch_std.clamp(1e-4, 1e6)

        # Update layers
        self.mean.data = batch_mean
        self.square_mean.data = batch_square_mean
        self.std.data = batch_std

    def apply(self, x, use_mean=True, shape=None):
        if shape is None: shape = tuple((len(x.shape)-1)*[1]+[-1])
        use_mean = self.use_mean * use_mean
        if use_mean: x = x - self.mean.view(shape)
        if self.use_std: x = x / self.std.view(shape)
        return x

    def remove(self, x, use_mean=True, shape=None):
        if shape is None: shape = tuple((len(x.shape)-1)*[1]+[-1])
        use_mean = self.use_mean * use_mean
        if self.use_std: x = x * self.std.view(shape)
        if use_mean: x = x + self.mean.view(shape)
        return x
    

class PopArtStandardization(ArtStandardization):
    "https://arxiv.org/pdf/1809.04474"
    def __init__(self, *layers, splits=None, segments=None, pre=False, dim=None, **kwargs):
        # Pre: Standardization for inputs, rather than outputs
        # Parameters
        self.layers = layers  # List of lists of layer groups
        self.splits = splits  # List of slices for each layer group, splits mean and std across layers
        self.segments = segments  # List of lists of segments for each layer in a layer group, splits a layer
        self.pre = pre

        # Automatically detect dim
        if dim is None:
            if pre:
                dim = sum(
                    l[0].weight[
                        :, segments[i][0]
                        if self.segments is not None and self.segments[i][0] is not None
                        else slice(None)]
                    .shape[1]
                    for i, l in enumerate(layers))
            else:
                dim = sum(
                    l[0].weight[
                        segments[i][0]
                        if self.segments is not None and self.segments[i][0] is not None
                        else slice(None)]
                    .shape[0]
                    for i, l in enumerate(layers))

        # Super
        super().__init__(dim=dim, **kwargs)

    def update(self, x):
        # Perform update
        prev_mean = self.mean.clone()
        prev_std = self.std.clone()
        super().update(x)

        for i, layers in enumerate(self.layers):
            split = self.splits[i] if self.splits is not None else None
            segments = self.segments[i] if self.segments is not None else None
            for j, layer in enumerate(layers):
                segment = segments[j] if segments is not None else None
                # Get previous
                mu = prev_mean
                sigma = prev_std
                mu_prime = self.mean if self.use_mean else mu
                sigma_prime = self.std if self.use_std else sigma
                if self.splits is not None:
                    mu, sigma, mu_prime, sigma_prime = mu[..., split], sigma[..., split], mu_prime[..., split], sigma_prime[..., split]

                # POP - Preserving Outputs Precisely (Original)
                if not self.pre:
                    # UNIT TEST (16 input, 64 out)
                    # ## Initialize
                    # X = torch.rand((1, 16), device='cuda')
                    # stand = celltrip.policy.PopArtStandardization(policy.actor_critic.self_pos_embed, dim=(64,)).cuda()
                    # logits = policy.actor_critic.self_pos_embed(X)
                    # print(stand.remove(logits))
                    # # print(logits)
                    # ## Update
                    # for _ in range(1_000): stand.update(logits)
                    # logits = policy.actor_critic.self_pos_embed(X)
                    # print(stand.remove(logits))  # Should be same
                    # print(logits)  # Should be different
                    pop_layer(layer, (mu, sigma), (mu_prime, sigma_prime), segment=segment)

                # POP for inputs (PIP)
                if self.pre:
                    # UNIT TEST (16 input, 64 out)
                    # ## Initialize
                    # X = torch.rand((1, 16), device='cuda')
                    # stand = celltrip.policy.PopArtStandardization(policy.actor_critic.self_pos_embed, pre=True, dim=(16,)).cuda()
                    # print(stand.apply(X))
                    # print(policy.actor_critic.self_pos_embed(stand.apply(X)))
                    # ## Update
                    # for _ in range(1_000): stand.update(X)
                    # print(stand.apply(X))  # Different than 1st
                    # print(policy.actor_critic.self_pos_embed(stand.apply(X)))  # Same as 2nd
                    pip_layer(layer, (mu, sigma), (mu_prime, sigma_prime), segment=segment)


def pop_layer(layer, mu_sigma, mu_sigma_prime, segment=None):
    # Defaults
    if segment is None: segment = slice(None)
    # Execute
    mu, sigma = mu_sigma
    mu_prime, sigma_prime = mu_sigma_prime
    layer.weight.data[segment] = layer.weight[segment] * sigma.unsqueeze(-1) / sigma_prime.unsqueeze(-1)
    layer.bias.data[segment] = (layer.bias[segment] * sigma + mu - mu_prime) / sigma_prime

def pip_layer(layer, mu_sigma, mu_sigma_prime, segment=None):
    # Defaults
    if segment is None: segment = slice(None)
    # Execute
    mu, sigma = mu_sigma
    mu_prime, sigma_prime = mu_sigma_prime
    layer.weight.data[:, segment] = layer.weight[:, segment] * sigma_prime.unsqueeze(0) / sigma.unsqueeze(0)
    layer.bias.data = layer.bias + torch.matmul((mu_prime-mu)/sigma, layer.weight[:, segment].T)


class BufferStandardization(nn.Module):
    def __init__(self, dim=(), buffer_size=1_000):
        super().__init__()

        self.buffer_size = buffer_size
        self.buffer = nn.Parameter(torch.zeros((buffer_size, *dim)), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim), requires_grad=False)

    def update(self, x):
        "First dimension must be batch"
        self.buffer.data[:-x.shape[0]] = self.buffer[x.shape[0]:].clone()
        self.buffer.data[-x.shape[0]:] = x
        self.mean.data = self.buffer.mean(keepdim=True, dim=0)
        self.std.data = self.buffer.std(keepdim=True, dim=0)

    def apply(self, x, mean=True, std=True):
        if mean: x = x - self.mean
        if std: x = x / self.std
        return x

    def remove(self, x, mean=True, std=True):
        if std: x = x * self.std
        if mean: x = x + self.mean
        return x
    

# https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py  # TODO (rewrite RBF and MMDLoss)
# class RBF(nn.Module):
#     def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
#         super().__init__()
#         self.bandwidth_multipliers = nn.Parameter(
#             mul_factor ** (torch.arange(n_kernels) - n_kernels // 2),
#             requires_grad=False)
#         self.bandwidth = bandwidth

#     def get_bandwidth(self, L2_distances):
#         if self.bandwidth is None:
#             n_samples = L2_distances.shape[0]
#             return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
#         return self.bandwidth

#     def forward(self, X):
#         L2_distances = torch.cdist(X, X) ** 2
#         return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


# class MMDLoss(nn.Module):
#     def __init__(self, kernel=RBF):
#         super().__init__()
#         self.kernel = kernel()

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))
#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY


class PinningNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        # Running
        epochs=5,  # Higher epochs needed for more complex model
        epoch_size=2**10,  # 1024,
        batch_size=2**6,  # 64
        # Standardization
        standardization_beta=3e-4,
        extra_first_layers=[],
        # Architecture
        spatial=False,
        # Matching
        activation=nn.PReLU,
        betas=(.99, .99),
        lr=3e-4,
        weight_decay=1e-5,
        lr_iters=None,
        lr_gamma=1,
        dropout=0.,
        **kwargs):
        # Init
        super().__init__()

        # Arguments
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim is None: hidden_dim = 2*input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.extra_first_layers = extra_first_layers
        self.spatial = spatial

        # Create MLP
        self.mlp = nn.Sequential(
            # Basic
            nn.Linear(input_dim, hidden_dim),
            activation(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),

            # Basic mu logvar
            # nn.Linear(input_dim, hidden_dim),
            # activation(), nn.Dropout(dropout),
            # nn.Linear(hidden_dim, 2*output_dim),

            # Complex
            # nn.Linear(input_dim, hidden_dim),
            # activation(), nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), activation(), nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # activation(), nn.Dropout(dropout),
            # nn.Linear(hidden_dim, output_dim),
        )
        self.first_layer = self.mlp[0]
        self.last_layer = self.mlp[-1]
        self.forward = self.forward_mlp
        
        # Optimizer and Scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), betas=betas, lr=lr, weight_decay=weight_decay, eps=1e-5)
        # self.optimizer = torch.optim.AdamW(self.parameters())
        if lr_iters is not None: self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=lr_iters)
        else: self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Loss
        # self.mmd_loss = MMDLoss()

        # Standardization
        # self.input_standardization = PopArtStandardization(
        #     [self.first_layer] + self.extra_first_layers,
        #     segments=[[None] + [slice(self.input_dim) for _ in range(len(self.extra_first_layers))]],
        #     beta=standardization_beta, pre=True, use_mean=False)
        self.input_standardization = PopArtStandardization(
            [self.first_layer], beta=standardization_beta, pre=True)
        # self.input_standardization = PopArtStandardization(
        #     self.extra_first_layers,
        #     segments=[[slice(self.input_dim) for _ in range(len(self.extra_first_layers))]],
        #     beta=standardization_beta, pre=True)
        # self.input_standardization = PopArtStandardization(
        #     dim=input_dim, beta=standardization_beta, pre=True)
        self.output_standardization = PopArtStandardization(
            [self.last_layer],
            segments=[[slice(0, self.output_dim), slice(self.output_dim, -1)]],
            beta=standardization_beta, dim=self.output_dim)

    def forward_mlp(self, X, Y=None, input_standardization=True, output_standardization=False, return_logvar=False):
        # Input standardization
        if input_standardization: X = self.input_standardization.apply(X)

        # Calculation
        logit = self.mlp(X)
        mu = logit
        # mu, logvar = logit[..., :self.output_dim], logit[..., self.output_dim:]

        # Output standardization
        if not output_standardization:
            mu = self.output_standardization.remove(mu)
            # logvar = self.output_standardization.remove(logvar.exp().std(), use_mean=False).square().log()  # Not the most efficient way, could just square log the std

        # Transform if needed
        if self.spatial and Y is not None:  # Might want to cache at some point
            R, t = _utility.processing.solve_rot_trans(mu, Y)
            mu = torch.matmul(mu, R) + t

        # if return_logvar: return mu, logvar
        return mu
    
    # def forward_att(self, q, k, v, input_standardization=True, output_standardization=False):
    #     pass
        
    # def forward(self, X, return_logits=False):
    #     embedded, mu, logvar = self.encode(X, return_logits=True)
    #     imputed = self.decode(embedded)
    #     if not return_logits: return imputed
    #     else: return imputed, embedded, mu, logvar

    # def encode(self, X, return_logits=False):
    #     logits = self.encoder(X)
    #     mu, logvar = logits[..., :self.hidden_dim], logits[..., self.hidden_dim:]
    #     logvar = logvar.clamp(min=-10, max=10)
    #     dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logvar/2) if self.training else 1e-7)
    #     dist = torch.distributions.Normal(loc=mu, scale=1e-7)
    #     embedded = dist.sample()
    #     if return_logits: return embedded, mu, logvar
    #     else: return embedded

    # def decode(self, X):
    #     return self.decoder(X)

    def compute_loss(self, Y_pred, Y_true):  # , logvar
        # Archived
        # KLD_loss = -.5 * (1 + torch.log(X_batch.var(dim=0)) - X_batch.mean(dim=0).square() - X_batch.var(dim=0)).mean(dim=-1)
        # STD_loss = (Y_pred.std(dim=0) - Y_batch.std(dim=0)).square().mean(dim=-1)
        # MMD_loss = self.mmd_loss(Y_true, Y_pred)
        # MAE_loss = (Y_true - Y_pred).abs().mean(dim=-1).mean(dim=0)

        # In-use
        MSE_loss = (Y_true - Y_pred).square().mean(dim=-1)  # Reconstruction, also doesn't care about important features when standardized
        # PAIR_loss = (torch.cdist(Y_pred, Y_pred) - torch.cdist(Y_true, Y_true)).square().mean(dim=-1)
        # NLL_loss = .5 * ((Y_true - Y_pred).square() / logvar.exp() + logvar).mean(dim=-1)  # NLL, doesn't account for covariance
        # if np.random.rand() < .001:
        #     print(Y_pred.shape)
        #     print(Y_true.shape)
        #     print(MSE_loss.shape)
        #     print(PAIR_loss.shape)
        #     print(MSE_loss[:5])
        #     print(PAIR_loss[:5])
        #     print()
        # loss = .5*MSE_loss + .5*PAIR_loss
        loss = MSE_loss
        return loss
    
    def update(self, states, target_modality, world_size=None):
        # Parameters
        if world_size is None: world_size = get_world_size('learners')

        # Fallback for no updating modules
        if world_size == 0: return torch.nan

        # Batch calculation
        epoch_size = np.ceil(self.epoch_size / world_size).astype(int)
        batch_size = np.ceil(self.batch_size / world_size).astype(int)
        batches = max(1, epoch_size // batch_size)

        if states is not None:
            # Prepare data
            valid_mask = states[2].sum(dim=-1) <= 1
            idx_to_sample = torch.argwhere(valid_mask)[:, 0]
            num_samples = valid_mask.sum().cpu().item()

            # Spatial data cache
            # transform_cache = {}

            # Train
            for epoch in range(self.epochs):
                # Randomly sample
                # epoch_idx = torch.randperm(num_samples)
                epoch_idx = np.random.choice(num_samples, epoch_size, replace=num_samples < epoch_size)
                epoch_loss = 0
                for batch in range(batches):
                    # Subsample
                    batch_idx = epoch_idx[batch*batch_size:(batch+1)*batch_size]
                    X_batch = states[0][states[2].sum(dim=-1) <= 1][batch_idx]
                    Y_batch = target_modality[states[2].sum(dim=-1) <= 1][batch_idx]
                    # X_stand = self.input_standardization.apply(X_batch)  # Intentional double-application of std to simulate actual std

                    # Compute normal prediction
                    Y_pred = self(X_batch)
                    # Y_pred, logvar = self(X_batch, return_logvar=True)
                    # Y_pred = self(X_stand)

                    # Transform if needed
                    if self.spatial:
                        # Spatial data cache
                        transform_cache = {}
                        batch_sample_idx = idx_to_sample[batch_idx]
                        for sidx in torch.unique(batch_sample_idx):
                            # Calculate unknown transforms using all available data
                            if sidx not in transform_cache:
                                with torch.no_grad():  # Try without this too, but need to move cache inwards (CAUSES ERROR WITHOUT)
                                    X_sample = self(states[0][sidx][valid_mask[sidx]])
                                Y_sample = target_modality[sidx][valid_mask[sidx]]
                                transform_cache[sidx.cpu().item()] = _utility.processing.solve_rot_trans(X_sample, Y_sample)

                        # Stack transforms
                        R, t = [torch.stack([transform_cache[sidx.cpu().item()][i] for sidx in batch_sample_idx], dim=0) for i in range(2)]  # Problem line
                        Y_pred = Y_pred.unsqueeze(dim=-2)
                        Y_pred = torch.matmul(Y_pred, R) + t
                        Y_pred = Y_pred.squeeze(dim=-2)
                        # logvar = logvar.unsqueeze(dim=-2)
                        # logvar = torch.matmul(logvar, R)  # TODO: Confirm correct
                        # logvar = logvar.squeeze(dim=-2)

                    # Losses
                    loss = self.compute_loss(Y_pred, Y_batch).mean()
                    # loss = self.compute_loss(Y_pred, logvar, Y_batch).mean()

                    # Compute VAE prediction
                    # imputed, embedded, mu, logvar = self(X_batch, return_logits=True)
                    # reconstruction_loss = (Y_batch - imputed).square().mean(dim=-1)
                    # kld_loss = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean(dim=-1)
                    # loss = (reconstruction_loss + kld_loss).mean(dim=0)

                    # Compute loss and step
                    epoch_loss += loss.detach() / batches
                    loss.backward()
                    synchronize(self, 'learners', grad=False, override_world_size=world_size)  # Synchronize
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update standardization
                    self.input_standardization.update(X_batch)
                    self.output_standardization.update(Y_batch)

        # Fallback for no samples
        else:
            # Main synchronizations
            for _ in range(self.epochs*batches):
                zero_parameters(self.parameters())  # Zero all parameters, very sketchy (non-grad sync)
                synchronize(self, 'learners', grad=False, override_world_size=world_size)

        # Iterate scheduler and return
        self.scheduler.step()
        return epoch_loss.cpu().item() if states is not None else torch.nan


### Training classes
def create_agent_from_env(env, pinning_modal_dims='auto', pinning_spatial=None, **kwargs):
    # Defaults
    if pinning_spatial is None: pinning_spatial = []
    if pinning_modal_dims == 'auto':
        # pinning_modal_dims
        pinning_modal_dims = np.array([m.shape[1] for m in env.get_modalities()])[env.target_modalities]
        # pinning_spatial
        new_pinning_spatial = [i for i, tm in enumerate(env.target_modalities) if tm in pinning_spatial]
        assert len(pinning_spatial) == len(new_pinning_spatial), f'Some modalities not found as targets in `pinning_spatial`'
        pinning_spatial = new_pinning_spatial

    # Return PPO instance
    return PPO(
        2*env.dim,
        # np.array(env.dataloader.modal_dims)[env.input_modalities],
        np.array([m.shape[1] for m in env.get_modalities()])[env.input_modalities],
        env.dim,
        pinning_dim=env.dim,
        pinning_modal_dims=pinning_modal_dims,
        pinning_spatial=pinning_spatial,
        discrete=env.discrete,
        **kwargs)


class PPO(nn.Module):
    def __init__(
            self,
            positional_dim,
            modal_dims,
            output_dim,
            # Pinning
            pinning_dim=None,
            pinning_modal_dims=None,  # If not passed, doesn't perform pinning
            pinning_spatial=[],
            # Model Parameters
            # model=EntitySelfAttention, # sample_strategy 'random-proximity'
            model=EntitySelfAttentionLite,  # sample_strategy None
            # Forward
            log_std_init=0,
            forward_batch_size=1_000,  # int(5e4),
            vision_size=1_000,  # torch.inf,
            # sample_strategy='random-proximity',
            sample_strategy=None,  # NOTE: Not used in Lite model
            sample_dim=None,
            reproducible_strategy='mean',
            # Weights
            epsilon_ppo=.2,
            epsilon_critic=torch.inf,
            critic_weight=1.,
            entropy_weight=1e-3,
            kl_beta_init=0.,
            kl_beta_increment=(.5, 2),
            kl_target=.03,
            kl_early_stop=False,
            grad_clip=.5,
            # Optimizers
            # log_std_lr=3e-4,
            # actor_lr=3e-4,
            # critic_lr=3e-4,
            lr=3e-4,
            standardization_beta=3e-3,
            weight_decay=1e-5,
            betas=(.99, .99),
            lr_iters=None,
            lr_gamma=1.,
            # Backward
            update_iterations=5,
            sync_iterations=1,
            pool_size=torch.inf,
            epoch_size=100_000,
            batch_size=10_000,  # https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?params=/context/etd_projects/article/1972/&path_info=park_inhee.pdf
            minibatch_size=torch.inf,
            minibatch_memories=1_000_000,  # 1M, 250k for extra hidden layer
            # load_level='minibatch',  # TODO: Allow for loading at batch with compression
            # cast_level='minibatch',
            actor_critic_kwargs={},
            pinning_kwargs={},
            **kwargs,
    ):
        super().__init__()

        # Parameters
        self.positional_dim = positional_dim
        self.modal_dims = modal_dims
        self.output_dim = output_dim

        # Pinning
        self.pinning_dim = pinning_dim
        self.pinning_modal_dims = pinning_modal_dims
        self.pinning_spatial = pinning_spatial

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
        self.split_kwargs = {
            'max_nodes': vision_size,
            'sample_strategy': sample_strategy,
            'reproducible_strategy': reproducible_strategy,
            'sample_dim': sample_dim}
        self.update_iterations = update_iterations
        self.sync_iterations = sync_iterations
        self.pool_size = pool_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.minibatch_memories = minibatch_memories

        # New policy
        self.actor_critic = model(positional_dim, modal_dims, output_dim, log_std_init=log_std_init, **actor_critic_kwargs)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), betas=betas, lr=lr, weight_decay=weight_decay, eps=1e-5)
        if lr_iters is not None: self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=lr_iters)
        else: self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)

        # Old policy
        self.actor_critic_old = model(positional_dim, modal_dims, output_dim, log_std_init=log_std_init, **actor_critic_kwargs)
        self.optimizer_old = torch.optim.Adam(self.actor_critic_old.parameters(), betas=betas, lr=lr, weight_decay=weight_decay, eps=1e-5)

        # Non-grad params
        self.policy_iteration = nn.Parameter(torch.tensor(0.), requires_grad=False)

        # PopArt
        self.input_standardization = PopArtStandardization(
            self.actor_critic.input_pos_layers, self.actor_critic.input_feat_layers,
            splits=(slice(0, self.positional_dim), slice(self.positional_dim, self.positional_dim+sum(self.modal_dims))), pre=True, beta=standardization_beta)
        self.return_standardization = PopArtStandardization(self.actor_critic.output_critic_layers, beta=standardization_beta)

        # Pinning (Assumes that the positional dim is pos+vel)
        if self.pinning_modal_dims is not None:
            self.pinning_dim = int(positional_dim/2) if self.pinning_dim is None else self.pinning_dim
            self.pinning = nn.ModuleList([
                PinningNN(
                    self.pinning_dim, pinning_modal_dim, spatial=(i in self.pinning_spatial),
                    betas=betas, lr=lr, weight_decay=weight_decay,
                    lr_iters=lr_iters, lr_gamma=lr_gamma,
                    extra_first_layers=self.actor_critic.input_pos_layers,
                    standardization_beta=standardization_beta, **pinning_kwargs)
                for i, pinning_modal_dim in enumerate(self.pinning_modal_dims)])
        else: self.pinning = None

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

    def get_policy_iteration(self):
        return int(self.policy_iteration.item())
    
    def set_vision_size(self, vision_size):
        # NOTE: Does not currently update memory object
        self.split_kwargs['max_nodes'] = vision_size
    
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
        name = 'CellTRIP' if name is None else name

        # Get all vars in order
        fname = os.path.join(directory, f'{name}-{int(self.policy_iteration.item()):0>4}.weights')
        policy_state = _utility.general.get_policy_state(self)

        # Save
        with _utility.general.open_s3_or_local(fname, 'wb') as f: torch.save(policy_state, f)
        # NOTE: Will no longer do `os.makedirs(directory, exist_ok=True)` for local
        return fname

    def load_checkpoint(self, fname, **kwargs):  # Can pass `strict=False` to ignore missing entries
        # Get from fname
        with _utility.general.open_s3_or_local(fname, 'rb') as f:
            policy_state = torch.load(f, map_location=self.policy_iteration.device)

        # Load policy
        _utility.general.set_policy_state(self, policy_state, **kwargs)
        return self

    def forward(
        self, compressed_state, *,
        keys=None, memory=None, forward_batch_size=None, terminal=False,
        feature_embeds=None, return_feature_embeds=False):
        # NOTE: `feature_embeds` will not re-randomize vision if applicable to `split_state` (i.e. do not use non-lite model with vision culling and feature embed caching)
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
            self_idx = np.arange(start_idx, min(start_idx+forward_batch_size, compressed_state.shape[0]))
            state = _utility.processing.split_state(
                self.input_standardization.apply(compressed_state),
                idx=self_idx,
                **self.split_kwargs)
            feature_embeds_arg_use = [(sfe[self_idx], nfe) for sfe, nfe in feature_embeds_arg] if feature_embeds_arg is not None else None
            if not terminal:
                action_sub, action_log_sub, state_val_sub, feature_embeds_sub = self.actor_critic(
                    *state, critic=True, feature_embeds=feature_embeds_arg_use, return_feature_embeds=True)
                action = torch.concat((action, action_sub), dim=0)
                action_log = torch.concat((action_log, action_log_sub), dim=0)
            else: state_val_sub, feature_embeds_sub = self.actor_critic(
                *state, actor=False, critic=True, feature_embeds=feature_embeds_arg_use, return_feature_embeds=True)
            state_val = torch.concat((state_val, state_val_sub), dim=0)
            if construct_feature_embeds:
                if feature_embeds is None: feature_embeds = feature_embeds_sub
                else:
                    # feature_embeds = [
                    #     tuple(torch.concat((feature_embeds[i][j], t)) for j, t in enumerate(feat_tensors))
                    #     for i, feat_tensors in enumerate(feature_embeds_sub)]
                    for i in range(len(feature_embeds)):
                        # NOTE: 0 is the self embeddings, which are the only ones subset in the Lite model
                        # feature_embeds[i][0] = torch.concat((feature_embeds[i][0], feature_embeds_sub[i][0]))
                        assert (feature_embeds[i][1] == feature_embeds_sub[i][1]).all(), 'Unexpected node embedding changes, make sure no vision culling is occurring.'
                        feature_embeds[i] = (torch.concat((feature_embeds[i][0], feature_embeds_sub[i][0])), feature_embeds[i][1])

        # Unstandardize state_val
        state_val = self.return_standardization.remove(state_val)
        
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
        ret = ()
        if not terminal: ret += action,
        if return_feature_embeds: ret += feature_embeds,
        # print(action)
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
        standardize_inputs=True,
        standardize_returns=True,
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
        # level_dict = {'pool': 0, 'epoch': 1, 'batch': 2, 'minibatch': 3}
        # load_level = level_dict[self.load_level]
        # cast_level = level_dict[self.cast_level]
        # assert cast_level >= load_level, 'Cannot cast without first loading'

        # Determine level sizes
        memory_size = len(memory)
        
        # Pool size
        pool_size = self.pool_size
        # if pool_size is not None:
        pool_size = min(pool_size, memory_size)
        
        # Epoch size
        epoch_size = self.epoch_size if not (self.epoch_size is None and pool_size is not None) else pool_size
        # if epoch_size is not None and pool_size is not None:
        epoch_size = int(min(epoch_size, pool_size))
        
        # Batch size
        batch_size = self.batch_size if not (self.batch_size is None and epoch_size is not None) else epoch_size
        # if batch_size is not None and epoch_size is not None:
        batch_size = int(min(batch_size, epoch_size))

        # Level adjustment
        denominator = get_world_size('learners') if sync_iterations == 1 else 1  # Adjust sizes if gradients synchronized across GPUs
        # if epoch_size is not None:
        if not np.isinf(self.epoch_size): epoch_size = np.ceil(epoch_size / denominator).astype(int)
        # if batch_size is not None:
        if not np.isinf(self.batch_size): batch_size = np.ceil(batch_size / denominator).astype(int)

        # Minibatch size
        minibatch_size = self.minibatch_size if not (self.minibatch_size is None and batch_size is not None) else batch_size
        if minibatch_size is not None and batch_size is not None: minibatch_size = int(min(minibatch_size, batch_size))

        # Load pool
        total_losses = defaultdict(lambda: [])
        total_statistics = defaultdict(lambda: [])
        pool_idx = np.random.choice(memory_size, pool_size, replace=False) if pool_size < memory_size else memory_size

        # Train
        iterations = 0; synchronized = True; escape = False
        while True:
            # Load epoch
            epoch_idx = np.random.choice(pool_idx, epoch_size, replace=False)  # Also shuffles
            batches = np.floor(epoch_size/batch_size).astype(int) if epoch_size is not None else 1  # Drop any smaller batches
            for batch_num in range(batches):
                # Load batch
                batch_losses = defaultdict(lambda: 0)
                batch_statistics = defaultdict(lambda: 0)
                batch_idx = epoch_idx[batch_num*batch_size:(batch_num+1)*batch_size]
                batch_state_vals_new = torch.zeros(0, device=self.policy_iteration.device)
                batch_inputs = torch.zeros(0, device=self.policy_iteration.device)
                batch_returns = torch.zeros(0, device=self.policy_iteration.device)
                minibatches = np.ceil(batch_size/minibatch_size).astype(int) if batch_size is not None else 1
                for minibatch_num in range(minibatches):
                    # Load minibatch
                    minibatch_idx = batch_idx[minibatch_num*minibatch_size:(minibatch_num+1)*minibatch_size]
                    minibatch_indices, minibatch_data = memory[minibatch_idx]

                    # Normalize advantages
                    minibatch_data['normalized_advantages'] = (minibatch_data['advantages'] - minibatch_data['advantages'].mean()) / (minibatch_data['advantages'].std() + 1e-8)

                    # Cast
                    minibatch_data = _utility.processing.dict_map_recursive_tensor_idx_to(minibatch_data, None, self.policy_iteration.device)

                    # Ministeps
                    minibatch_memories = self.minibatch_memories if self.minibatch_memories is not None else np.prod(minibatch_data['states'][1].shape[:-1])
                    ministep_size = np.maximum(np.floor(minibatch_memories / minibatch_data['states'][1].shape[1]), 1).astype(int)
                    ministeps = np.ceil(minibatch_data['states'][1].shape[0] / ministep_size).astype(int)
                    cumsum_indices = (minibatch_indices > -1).flatten().cumsum(0).reshape(minibatch_indices.shape)
                    proc_mems = 0

                    # Debug
                    # print(minibatch_memories)
                    # print(minibatch_data['states'][1].shape[1])
                    # print(ministep_size)
                    # print(minibatch_data['states'][1].shape[0])
                    # print()
                    
                    for ministep_num in range(ministeps):
                        # Subsample
                        double_idx = slice(ministep_num*ministep_size, (ministep_num+1)*ministep_size)
                        single_idx = cumsum_indices[double_idx]
                        first_true = int(minibatch_indices[double_idx][0, 0] > -1)
                        num_memories = single_idx.max() - single_idx.min() + first_true
                        single_idx = slice(single_idx.min() - first_true, single_idx.max())  # NOTE: No +1 here because cumsum is always one ahead
                        if num_memories == 0: continue
                        proc_mems += num_memories

                        # Get subset data
                        states = [s[double_idx] for s in minibatch_data['states']]
                        actions = minibatch_data['actions'][single_idx]
                        action_logs = minibatch_data['action_logs'][single_idx]
                        state_vals = minibatch_data['state_vals'][single_idx]
                        advantages = minibatch_data['advantages'][single_idx]
                        normalized_advantages = minibatch_data['normalized_advantages'][single_idx]
                        # rewards = minibatch_data['propagated_rewards'][single_idx]

                        # Perform backward
                        loss, loss_ppo, loss_critic, loss_entropy, loss_kl, state_vals_new = self.calculate_losses(
                            states, actions, action_logs, state_vals, advantages=advantages, normalized_advantages=normalized_advantages, rewards=None)

                        # Scale and calculate gradient
                        # accumulation_frac = minibatch_actual_size / batch_size
                        # accumulation_frac = minibatch_idx.shape[0] / batch_size
                        accumulation_frac = num_memories / batch_size
                        loss = loss * accumulation_frac
                        loss.backward()  # Longest computation

                        # Record required logs
                        batch_inputs = torch.cat((batch_inputs, states[0].view(-1, states[0].shape[-1])), dim=0)  # Only use first to save memory
                        batch_returns = torch.cat((batch_returns, advantages+state_vals), dim=0)
                        batch_state_vals_new = torch.cat((batch_state_vals_new, state_vals_new), dim=0)

                        # Scale and record
                        batch_losses['Total'] += loss.detach()
                        batch_losses['PPO'] += loss_ppo.detach().mean() * accumulation_frac
                        batch_losses['critic'] += loss_critic.detach().mean() * accumulation_frac
                        batch_losses['entropy'] += loss_entropy.detach().mean() * accumulation_frac
                        batch_losses['KL'] += loss_kl.detach().mean() * accumulation_frac

                # Calculate explained variance
                normalized_returns = self.return_standardization.apply(batch_returns)
                exp_var = (1- (normalized_returns - batch_state_vals_new).var() / normalized_returns.var()).clamp(min=-1)

                # Statistics
                batch_statistics['Moving Return Mean'] += self.return_standardization.mean.detach().mean()
                batch_statistics['Moving Return STD'] += self.return_standardization.std.detach().mean()
                batch_statistics['Return Mean'] += batch_returns.detach().mean() * accumulation_frac
                batch_statistics['Return STD'] += batch_returns.detach().std() * accumulation_frac
                batch_statistics['Moving Input Mean'] += self.input_standardization.mean.detach().mean()
                batch_statistics['Moving Input STD'] += self.input_standardization.std.detach().mean()
                batch_statistics['Input Mean'] += batch_inputs.detach().mean() * accumulation_frac
                batch_statistics['Input STD'] += batch_inputs.detach().std() * accumulation_frac
                # batch_statistics['Moving Reward Mean'] += self.reward_standardization.mean.mean() * accumulation_frac
                # batch_statistics['Moving Reward STD'] += self.reward_standardization.std.mean() * accumulation_frac
                batch_statistics['Explained Variance'] += exp_var.detach()
                # batch_statistics['Advantage Mean'] += advantages.detach().mean() * accumulation_frac
                # batch_statistics['Advantage STD'] += advantages.detach().std() * accumulation_frac
                # batch_statistics['Log STD'] += self.get_log_std() * accumulation_frac
                
                # Record
                for k, v in batch_losses.items(): total_losses[k].append(v)
                for k, v in batch_statistics.items(): total_statistics[k].append(v)

                # Synchronize GPU policies and step
                # NOTE: Synchronize gradients every batch if =1, else synchronize whole model
                # NOTE: =1 keeps optimizers in sync without need for whole-model synchronization
                if sync_iterations == 1: synchronize(self, 'learners', grad=False)  # Sync only grad
                if self.kl_early_stop and synchronized: self.copy_policy()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if sync_iterations != 1:
                    # Synchronize for offsets
                    sync_loop = (iterations) % sync_iterations == 0
                    last_epoch = iterations == update_iterations
                    if use_collective and (sync_loop or last_epoch):
                        synchronize(self, 'learners')
                        synchronized = True
                    else: synchronized = False

                # Update moving return mean
                if standardize_inputs:
                    self.input_standardization.update(batch_inputs)
                if standardize_returns:
                    self.return_standardization.update(batch_returns)
                    synchronize(self, 'learners', sync_list=self.return_standardization.parameters())

                # Update KL beta
                # NOTE: Same as Torch KLPENPPOLoss implementation
                if self.kl_early_stop or self.kl_beta != 0:
                    loss_kl_mean = loss_kl.detach().mean()
                    synchronize(self, 'learners', sync_list=[loss_kl_mean])
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
                    + f' :: '
                    + f', '.join([f'{k} ({np.mean([v.item() for v in vl[-batches:]]):.5f})' for k, vl in total_statistics.items()]))

            # Break
            if escape: break

        # Update scheduler
        self.scheduler.step()

        # Train pinning
        states, target_modalities = memory.get_terminal_pairs()
        has_memories = torch.tensor(0., device=self.policy_iteration.device)
        if states is not None:
            states[0], states[1] = states[0][..., :self.pinning_dim], states[1][..., :self.pinning_dim]
            has_memories += 1
        synchronize(self, 'learners', sync_list=[has_memories], override_world_size=1)
        has_memories = int(has_memories.cpu().item())
        running_feat = 0; pinning_losses = {}
        for i, (pinning, pinning_modal_dim) in enumerate(zip(self.pinning, self.pinning_modal_dims)):
            if states is None: pinning.update(None, None, world_size=has_memories)
            else:
                X_sub = states[0][states[2].sum(dim=2) <= 1].to(self.policy_iteration.device)
                Y_sub = target_modalities[..., running_feat:running_feat+pinning_modal_dim].to(self.policy_iteration.device)
                pinning_losses[f'Pinning Mean'] = X_sub.mean(dim=0).mean().cpu().item()  # Calculated multiple times, but no big performance hit
                pinning_losses[f'Pinning Abs Mean'] = X_sub.mean(dim=0).abs().mean().cpu().item()
                pinning_losses[f'Pinning STD'] = X_sub.std(dim=0).mean().cpu().item()
                pinning_losses[f'Pinning Loss {i}'] = pinning.update(
                    states, Y_sub, world_size=has_memories)
            running_feat += pinning_modal_dim
        if target_modalities is not None:
            assert running_feat == target_modalities.shape[-1], (
                f'`pinning_modal_dims` sum ({running_feat}) does not match target modality combined length ({target_modalities.shape[1]})')

        # Update records
        self.policy_iteration += 1
        self.copy_policy()
        # Return
        total_losses_ret = {k: np.mean([v.item() for v in vl]) for k, vl in total_losses.items()}
        total_losses_ret = {**total_losses_ret, **pinning_losses}  # TODO: Maybe revise
        total_statistics_ret = {k: np.mean([v.item() for v in vl]) for k, vl in total_statistics.items()}
        return (
            iterations,
            total_losses_ret,
            total_statistics_ret)

    def calculate_losses(
        self,
        states,
        actions,
        action_logs,
        state_vals,
        advantages=None,
        normalized_advantages=None,
        rewards=None):
        # TODO: Maybe implement PFO https://github.com/CLAIRE-Labo/no-representation-no-trust
        if advantages is not None:
            # Get inferred rewards
            rewards = advantages + state_vals
        elif rewards is not None:
            # Get advantages
            advantages = rewards - state_vals
        # Get normalized advantages
        if normalized_advantages is None:
            advantages_mean, advantages_std = advantages.mean(), advantages.std() + 1e-8
            normalized_advantages = (advantages - advantages_mean) / advantages_std
        # Clip advantages for stability
        # normalized_advantages =  normalized_advantages.clamp(
        #     normalized_advantages.quantile(.05), normalized_advantages.quantile(.95))

        # Get normalized returns/rewards (sensitive to minibatch)
        # NOTE: Action STD explosion/instability points to a normalization and/or critic fitting issue
        #       or, possibly, the returns are too homogeneous - i.e. the problem is solved
        normalized_rewards = self.return_standardization.apply(rewards)  # , mean=False

        # Evaluate actions and states
        normalized_states = [self.input_standardization.apply(states[0]), self.input_standardization.apply(states[1]), states[2]]
        action_logs_new, dist_entropy, state_vals_new = self.actor_critic(*normalized_states, action=actions, entropy=True, critic=True)
        # action_logs_new = action_logs_new.clamp(-20, 0)
        
        # Calculate PPO loss
        log_ratios = action_logs_new - action_logs
        # log_ratios = log_ratios.clamp(-20, 2)
        ratios = log_ratios.exp()
        unclipped_ppo = ratios * normalized_advantages
        clipped_ppo = ratios.clamp(1-self.epsilon_ppo, 1+self.epsilon_ppo) * normalized_advantages
        loss_ppo = -torch.min(unclipped_ppo, clipped_ppo)

        # Calculate KL divergence
        # NOTE: A bit odd when it comes to replay
        # Discrete
        # loss_kl = F.kl_div(action_logs, action_logs_new, reduction='batchmean', log_target=True)
        # loss_kl = ((action_logs - action_logs_new)  # * action_logs.exp()).sum(-1)  # Approximation
        # Continuous (http://joschu.net/blog/kl-approx.html)
        loss_kl = (log_ratios.exp() - 1) - log_ratios
        # Mask and scale where needed (for replay)
        # loss_kl[~new_memories] = 0
        # loss_kl = loss_kl * loss_kl.shape[0] / new_memories.sum()

        # Calculate critic loss
        criteria = F.smooth_l1_loss
        # criteria = F.mse_loss
        unclipped_critic = criteria(state_vals_new, normalized_rewards)
        clipped_state_vals_new = torch.clamp(state_vals_new, state_vals-self.epsilon_critic, state_vals+self.epsilon_critic)
        clipped_critic = criteria(clipped_state_vals_new, normalized_rewards, reduction='none')
        loss_critic = torch.max(unclipped_critic, clipped_critic)

        # Calculate entropy bonus
        # NOTE: Not included in training grad if action_std is constant
        # dist_entropy = -action_logs_new  # Approximation
        loss_entropy = -dist_entropy

        # Construct final loss
        loss = (
            loss_ppo
            + self.critic_weight * loss_critic
            + self.entropy_weight * loss_entropy
            + self.kl_beta * loss_kl)
        loss = loss.mean()

        return loss, loss_ppo, loss_critic, loss_entropy, loss_kl, state_vals_new


def get_world_size(group='default', warn=False):
    try:
        world_size = col.get_collective_group_size(group)
        if world_size == -1: raise RuntimeError
        return world_size
    except:
        if warn: warnings.warn(f'No group "{group}" found.')
        return 1
    

def get_rank(group='default', warn=False):
    try:
        rank = col.get_rank(group)
        if rank == -1: raise RuntimeError
        return rank
    except:
        if warn: warnings.warn(f'No group "{group}" found.')
        return 0


def synchronize(module, group='default', sync_list=None, grad=False, src_rank=0, override_world_size=None, broadcast=None, allreduce=None):
    # Defaults
    if broadcast is None: broadcast = False
    if allreduce is None: allreduce = not broadcast

    # Collective operations
    world_size = get_world_size(group)
    if world_size == 1: return

    # Sync
    sync_list = module.parameters() if sync_list is None else sync_list
    with torch.no_grad():
        for w in sync_list:  # zip(module.state_dict(), module.parameters())
            if grad: w = w.grad  # No in-place modification here
            if w is None: continue
            if w.dtype == torch.long: continue
            if broadcast: col.broadcast(w, src_rank, group)
            if allreduce:
                col.allreduce(w, group)
                w /= world_size if override_world_size is None else override_world_size


def zero_parameters(parameters, factor=0):
    with torch.no_grad():
        for param in parameters:
            if param is None: continue
            if param.dtype == torch.long: continue
            param *= factor
