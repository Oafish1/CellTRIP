# %%
# TODO
# Check that rewards are normalized after (?) advantage
# Add FCL between raw features and the features appended to vector
# More efficient state storage (store modalities only once, etc.)
# Add option to not save states on forward for policy
# Make distance reward env-size and dataset-agnostic (i.e. spawn nodes in range 0-1 (or at origin), normalize dist per dataset (maybe by average dist))
# Try MMD-MA with all features

# Fix off-center positioning in large environments (kinda solved with post-centering?)
# Try using running average early stopping
# Save checkpoint models
# Try with 100s of nodes
# Try transfer learning with stages like stay at origin -> regular dist -> etc.
# Upload episode playback data to wandb
# Add parallel envs of different sizes, with different data to help generality

# %%
# %load_ext autoreload
# %autoreload 2
# %env WANDB_NOTEBOOK_NAME train.ipynb
# %env WANDB_SILENT true

# %%
from collections import defaultdict
import itertools
import os

import inept
import numpy as np
import pandas as pd
import torch
import wandb

# Set params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_FOLDER = os.path.join(os.path.abspath(''), '../data')
MODEL_FOLDER = os.path.join(os.path.abspath(''), 'temp/trained_models')

# Script arguments
# import sys
# arg1 = int(sys.argv[1])

# %%
# Original paper (pg 24)
# https://arxiv.org/pdf/1909.07528.pdf

# Original blog
# https://openai.com/research/emergent-tool-use

# Gym
# https://gymnasium.farama.org/

# Slides
# https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf

# PPO implementation
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L38

# Residual SA
# https://github.com/openai/multi-agent-emergence-environments/blob/bafaf1e11e6398624116761f91ae7c93b136f395/ma_policy/layers.py#L89

# %%
# Reproducibility
seed = 42
torch.manual_seed(seed)
if DEVICE == 'cuda': torch.cuda.manual_seed(seed)
np.random.seed(seed)

note_kwargs = {'seed': seed}

# %% [markdown]
# ### Create Environment

# %%
# MMD-MA data
num_nodes = 50
data_kwargs = {
    'dataset': 'MMD-MA',
    'num_nodes': num_nodes,
}
M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped1.txt'), delimiter='\t', header=None).to_numpy()
M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped2.txt'), delimiter='\t', header=None).to_numpy()
T1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type1.txt'), delimiter='\t', header=None).to_numpy()
T2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type2.txt'), delimiter='\t', header=None).to_numpy()
# Subsample
idx = np.random.choice(M1.shape[0], num_nodes, replace=False)
idx_1 = np.random.choice(M1.shape[1], 256, replace=False)
idx_2 = np.random.choice(M2.shape[1], 256, replace=False)
M1 = torch.tensor(M1[idx][:, idx_1], dtype=torch.float32, device=DEVICE)
M2 = torch.tensor(M2[idx][:, idx_2], dtype=torch.float32, device=DEVICE)
T1 = torch.tensor(T1[idx], dtype=torch.long, device=DEVICE)
T2 = torch.tensor(T2[idx], dtype=torch.long, device=DEVICE)
modalities = (M1, M2)

# Random data
# num_nodes = 20
# data_kwargs = {
#     'dataset': 'Random',
#     'num_nodes': num_nodes,
# }
# M1 = torch.rand((num_nodes, 8), device=DEVICE)
# M2 = torch.rand((num_nodes, 16), device=DEVICE)
# modalities = (M1, M2)

# Environment
# x, y, vx, vy
num_dims = 2
env_kwargs = {
    'dim': num_dims,
    'pos_bound': 2,
    'vel_bound': 1,
    'delta': .1,
    'reward_distance': 10,
    # 'reward_origin': 1,
    'penalty_bound': 1,
    'penalty_velocity': 1,
    'penalty_action': 1,
    'reward_distance_type': 'euclidean',
}
env = inept.environments.trajectory(*modalities, **env_kwargs, device=DEVICE)

# %% [markdown]
# ### Train Policy

# %%
# Tracking parameters
# Use `watch -d -n 0.5 nvidia-smi` to watch CUDA memory usage
# Use `top` to watch system memory usage
use_wandb = False

# Policy parameters
input_dims = 2*num_dims+sum([m.shape[1] for m in modalities])
memory_factor = 1  # High if large number of features
update_minibatch = int( 4e4 * (10 / num_nodes) / memory_factor )
update_max_batch = memory_factor * update_minibatch  # Only run one minibatch, empirically the benefit is minimal compared to time loss
policy_kwargs = {
    'num_features_per_node': input_dims,
    'output_dim': num_dims,
    'action_std_init': .6,
    'action_std_decay': .05,
    'action_std_min': .1,
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    'lr_gamma': 1,
    'update_minibatch': update_minibatch,  # Based on no minibatches needed with 10 nodes at 4k update timesteps
    'update_max_batch': update_max_batch,  # Try making larger, e.g. 20x minibatches
    'device': DEVICE,
}
policy = inept.models.PPO(**policy_kwargs)

# Training parameters
max_ep_timesteps = 2e2  # 2e2
max_timesteps = 1e6
update_timesteps = 4e3  # 20 * max_ep_timesteps
train_kwargs = {
    'max_ep_timesteps': max_ep_timesteps,
    'max_timesteps': max_timesteps,
    'update_timesteps': update_timesteps,
}

# Early stopping parameters
es_kwargs = {
    'buffer': 3 * int(update_timesteps / max_ep_timesteps),
    'delta': .01,
}
early_stopping = inept.utilities.EarlyStopping(**es_kwargs)

# Initialize wandb
if use_wandb: wandb.init(
    project='INEPT',
    config={
        **{'note/'+k:v for k, v in note_kwargs.items()},
        **{'data/'+k:v for k, v in data_kwargs.items()},
        **{'env/'+k:v for k, v in env_kwargs.items()},
        **{'policy/'+k:v for k, v in policy_kwargs.items()},
        **{'train/'+k:v for k, v in train_kwargs.items()},
        **{'es/'+k:v for k, v in es_kwargs.items()},
    },
)

# Initialize logging vars
torch.cuda.reset_peak_memory_stats()
timer = inept.utilities.time_logger(discard_first_sample=True)
timestep = 0; episode = 1

# CLI
print('Beginning training')
print(f'Subsampling {update_max_batch} states with minibatches of size {update_minibatch} from {int(update_timesteps * num_nodes)} total.')

# Simulation loop
while timestep < max_timesteps:
    # Reset environment
    env.reset()
    # env.reward_scales['reward_origin'] = episode / 1e3
    timer.log('Reset Environment')

    # Start episode
    ep_timestep = 0; ep_reward = 0; ep_itemized_reward = defaultdict(lambda: 0)
    while ep_timestep < max_ep_timesteps:
        with torch.no_grad():
            # Get current state
            state = env.get_state(include_modalities=True)

            # Get self features for each node
            self_entity = state

            # Get node features for each state
            idx = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
            for i, j in itertools.product(*[range(x) for x in idx.shape]):
                idx[i, j] = i!=j
            node_entities = state.unsqueeze(0).expand(num_nodes, *state.shape)
            node_entities = node_entities[idx].reshape(num_nodes, num_nodes-1, input_dims)
            timer.log('Environment Setup')

            # Get actions from policy
            actions = policy.act(self_entity, node_entities).detach()
            timer.log('Calculate Actions')

            # Step environment and get reward
            rewards, finished, itemized_rewards = env.step(actions, return_rewards=True)
            timer.log('Step Environment')

            # Record rewards
            for key in range(num_nodes):
                policy.memory.rewards.append(rewards[key].item())  # Could just add lists
                policy.memory.is_terminals.append(finished)
            ep_reward = ep_reward + rewards.cpu().mean()
            for k, v in itemized_rewards.items():
                ep_itemized_reward[k] += v.cpu().mean()
            timer.log('Record Rewards')

        # Iterate
        timestep += 1
        ep_timestep += 1

        # Update model
        if timestep % update_timesteps == 0:
            print(f'Updating model with average reward {np.mean(policy.memory.rewards)} on episode {episode} and timestep {timestep}', end='')
            policy.update()
            print(f' ({torch.cuda.max_memory_allocated() / 1024**3:.2f} GB CUDA)')
            torch.cuda.reset_peak_memory_stats()
            timer.log('Update Policy')

        # Escape if finished
        if finished: break

    # Upload stats
    ep_reward = (ep_reward / ep_timestep).item()
    if use_wandb:
        wandb.log({
            **{
            'episode': episode,
            'update': int(timestep / update_timesteps),
            'end_timestep': timestep,
            'average_reward': ep_reward,
            'action_std': policy.action_std,
            },
            **{'rewards/'+k: (v / ep_timestep).item() for k, v in ep_itemized_reward.items()},
        })
    timer.log('Record Stats')

    # Decay model std
    if early_stopping(ep_reward):
        # End if already at minimum
        if policy.action_std <= policy.action_std_min:
            print(f'Ending early on episode {episode} and timestep {timestep}')
            break

        # Decay and reset early stop
        policy.decay_action_std()
        early_stopping.reset()

        # CLI
        print(f'Decaying std to {policy.action_std} on episode {episode} and timestep {timestep}')
    timer.log('Early Stopping')

    # Iterate
    episode += 1

# CLI Timer
print()
timer.aggregate('sum')

# Save model
wgt_file = os.path.join(MODEL_FOLDER, 'policy.wgt')
torch.save(policy.state_dict(), wgt_file)  # Save just weights
if use_wandb: wandb.save(wgt_file)
mdl_file = os.path.join(MODEL_FOLDER, 'policy.mdl')
torch.save(policy, mdl_file)  # Save whole model
if use_wandb: wandb.save(mdl_file)

# Finish wandb
if use_wandb: wandb.finish()

# %%
