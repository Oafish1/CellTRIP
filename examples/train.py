
# %%
from collections import defaultdict
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
# ### Load Data

# %%
# Dataset loading
dataset_name = 'BrainChromatin'
if dataset_name == 'BrainChromatin':
    M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_rna_counts.tsv'), delimiter='\t', nrows=2_000).transpose()  # TODO: Raise number of features
    M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_atac_gene_activities.tsv'), delimiter='\t', nrows=2_000).transpose()  # TODO: Raise number of features
    M2 = M2.transpose()[M1.index].transpose()
    meta = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_cell_metadata.txt'), delimiter='\t')
    meta_names = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_cluster_names.txt'), delimiter='\t')
    meta_names = meta_names[meta_names['Assay'] == 'Multiome ATAC']
    meta = pd.merge(meta, meta_names, left_on='ATAC_cluster', right_on='Cluster.ID', how='left')
    meta.index = meta['Cell.ID']
    T1 = T2 = np.array(meta.transpose()[M1.index].transpose()['Cluster.Name'])
    F1, F2 = M1.columns, M2.columns
    M1, M2 = M1.to_numpy(), M2.to_numpy()

    del meta, meta_names

elif dataset_name == 'scGEM':
    M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/GeneExpression.txt'), delimiter=' ', header=None).to_numpy()
    M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/DNAmethylation.txt'), delimiter=' ', header=None).to_numpy()
    T1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/type1.txt'), delimiter=' ', header=None).to_numpy()
    T2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/type2.txt'), delimiter=' ', header=None).to_numpy()
    F1 = np.loadtxt(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/gex_names.txt'), dtype='str')
    F2 = np.loadtxt(os.path.join(DATA_FOLDER, 'UnionCom/scGEM/dm_names.txt'), dtype='str')

# MMD-MA data
elif dataset_name == 'MMD-MA':
    M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped1.txt'), delimiter='\t', header=None).to_numpy()
    M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped2.txt'), delimiter='\t', header=None).to_numpy()
    T1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type1.txt'), delimiter='\t', header=None).to_numpy()
    T2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type2.txt'), delimiter='\t', header=None).to_numpy()

# Random data
elif dataset_name == 'Random':
    num_nodes = 100
    M1 = torch.rand((num_nodes, 8), device=DEVICE)
    M2 = torch.rand((num_nodes, 16), device=DEVICE)

else: assert False, 'No matching dataset found.'

# Parameters
num_nodes = 20  # M1.shape[0]

# Modify data
M1, M2 = inept.utilities.normalize(M1, M2)  # Normalize
# M1, M2 = inept.utilities.pca_features(M1, M2, num_features=(16, 16))  # PCA features
M1, M2, T1, T2 = inept.utilities.subsample_nodes(M1, M2, T1, T2, num_nodes=num_nodes)  # Subsample nodes
# M1, M2 = inept.utilities.subsample_features(M1, M2, num_features=(16, 16))  # Subsample features

# Cast types
M1 = torch.tensor(M1, dtype=torch.float32, device=DEVICE)
M2 = torch.tensor(M2, dtype=torch.float32, device=DEVICE)
modalities = (M1,)  # , M2

# %% [markdown]
# ### Parameters

# %%
# Data parameters
data_kwargs = {
    'dataset': dataset_name,
    'num_nodes': num_nodes,
}

# Environment parameters
env_kwargs = {
    'dim': 2,  # x, y, vx, vy
    'pos_bound': 5,
    'pos_rand_bound': 1,
    'vel_bound': 1,
    'delta': .1,
    # 'reward_distance': 0,
    # 'reward_origin': 0,
    # 'penalty_bound': 0,
    # 'penalty_velocity': 0,
    # 'penalty_action': 0,
    'reward_distance_type': 'euclidean',
}

# Environment weight stages
stages_kwargs = {
    'env': (
        # Stage 0
        {'penalty_bound': 1},
        # Stage 1
        {'reward_origin': 1},
        # Stage 2
        {'penalty_velocity': 1, 'penalty_action': 1},
        # Stage 3
        {'reward_origin': 0, 'reward_distance': 1},
    ),
}

# Training parameters
train_kwargs = {
    'max_ep_timesteps': 1e3,  # Normal: 2e2
    'max_timesteps': 1e6,
    'update_timesteps': 5e3,  # Normal: 4e3
}

# Policy parameters
update_minibatch = int( 1e4 * (2000 / sum(M.shape[1] for M in modalities)) * (20 / data_kwargs["num_nodes"]) )  # Optimized for 1080 Ti
update_max_batch = int( 2e4 )
policy_kwargs = {
    # Main arguments
    'num_features_per_node': 2*env_kwargs['dim'],
    'modal_sizes': [M.shape[1] for M in modalities],
    'output_dim': env_kwargs['dim'],
    'action_std_init': .6,
    'action_std_decay': .05,
    'action_std_min': .1,
    'epochs': 80,
    'epsilon_clip': .2,
    'memory_gamma': .99,
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    'lr_gamma': 1,
    'update_minibatch': min( update_minibatch, update_max_batch ),  # If too high, the kernel will crash (can also often crash machine)
    'update_max_batch': update_max_batch,  # All memories: int(train_kwargs['update_timesteps'] * data_kwargs["num_nodes"])
    'device': DEVICE,
    # Layer arguments
    'embed_dim': 64,
    'feature_embed_dim': 32,
}

# Early stopping parameters
es_kwargs = {
    # Global parameters
    'method': 'average',
    'buffer': 6 * int(train_kwargs['update_timesteps'] / train_kwargs['max_ep_timesteps']),  # 6 training cycles
    'delta': .01,
    'decreasing': False,
    # `average` method parameters
    'window_size': 3 * int(train_kwargs['update_timesteps'] / train_kwargs['max_ep_timesteps']),  # 3 training cycles
}

# %% [markdown]
# ### Train Policy

# %%
# Tracking parameters
# Use `watch -d -n 0.5 nvidia-smi` to watch CUDA memory usage
# Use `top` to watch system memory usage
use_wandb = False

# Initialize classes
env = inept.environments.trajectory(*modalities, **env_kwargs, **stages_kwargs['env'][0], device=DEVICE)  # Set to first stage
policy = inept.models.PPO(**policy_kwargs)
early_stopping = inept.utilities.EarlyStopping(**es_kwargs)

# Initialize wandb
if use_wandb: wandb.init(
    project='INEPT',
    config={
        **{'note/'+k:v for k, v in note_kwargs.items()},
        **{'data/'+k:v for k, v in data_kwargs.items()},
        **{'env/'+k:v for k, v in env_kwargs.items()},
        **{'stages/'+k:v for k, v in stages_kwargs.items()},
        **{'policy/'+k:v for k, v in policy_kwargs.items()},
        **{'train/'+k:v for k, v in train_kwargs.items()},
        **{'es/'+k:v for k, v in es_kwargs.items()},
    },
)

# Initialize logging vars
torch.cuda.reset_peak_memory_stats()
timer = inept.utilities.time_logger(discard_first_sample=True)
timestep = 0; episode = 1; stage = 0

# CLI
print('Beginning training')
print(f'Subsampling {policy_kwargs["update_max_batch"]} states with minibatches of size {policy_kwargs["update_minibatch"]} from {int(train_kwargs["update_timesteps"] * data_kwargs["num_nodes"])} total.')

# Simulation loop
while timestep < train_kwargs['max_timesteps']:
    # Reset environment
    env.reset()
    timer.log('Reset Environment')

    # Start episode
    ep_timestep = 0; ep_reward = 0; ep_itemized_reward = defaultdict(lambda: 0)
    while ep_timestep < train_kwargs['max_ep_timesteps']:
        with torch.no_grad():
            # Get current state
            state = env.get_state(include_modalities=True)
            timer.log('Environment Setup')

            # Get actions from policy
            actions = policy.act_macro(state, keys=list(range(num_nodes))).detach()
            timer.log('Calculate Actions')

            # Step environment and get reward
            rewards, finished, itemized_rewards = env.step(actions, return_rewards=True)
            finished = finished or (ep_timestep == train_kwargs['max_ep_timesteps']-1)  # Maybe move logic inside env?
            timer.log('Step Environment')

            # Record rewards for policy
            policy.memory.record(
                rewards=rewards.cpu().tolist(),
                is_terminals=finished,
            )

            # Record rewards for logging
            ep_reward = ep_reward + rewards.cpu().mean()
            for k, v in itemized_rewards.items():
                ep_itemized_reward[k] += v.cpu().mean()
            timer.log('Record Rewards')

        # Iterate
        timestep += 1
        ep_timestep += 1

        # Update model
        if timestep % train_kwargs['update_timesteps'] == 0:
            # assert False
            print(f'Updating model with average reward {np.mean(policy.memory.storage["rewards"])} on episode {episode} and timestep {timestep}', end='')
            policy.update()
            print(f' ({torch.cuda.max_memory_allocated() / 1024**3:.2f} GB CUDA)')
            torch.cuda.reset_peak_memory_stats()
            timer.log('Update Policy')
            exit()

        # Escape if finished
        if finished: break

    # Upload stats
    ep_reward = (ep_reward / ep_timestep).item()
    update = int(timestep / train_kwargs['update_timesteps'])
    if use_wandb:
        wandb.log({
            **{
            # Measurements
            'end_timestep': timestep,
            'episode': episode,
            'update': update,
            'stage': stage,
            # Parameters
            'action_std': policy.action_std,
            # Outputs
            'average_reward': ep_reward,
            },
            **{'rewards/'+k: (v / ep_timestep).item() for k, v in ep_itemized_reward.items()},
        })
    timer.log('Record Stats')

    # Decay model std
    if early_stopping(ep_reward) or timestep >= train_kwargs['max_timesteps']:
        # Save model
        wgt_file = os.path.join(MODEL_FOLDER, f'policy_{stage:02}.wgt')
        torch.save(policy.state_dict(), wgt_file)  # Save just weights
        if use_wandb: wandb.save(wgt_file)
        mdl_file = os.path.join(MODEL_FOLDER, f'policy_{stage:02}.mdl')
        torch.save(policy, mdl_file)  # Save whole model
        if use_wandb: wandb.save(mdl_file)

        # End if maximum timesteps reached
        if timestep >= train_kwargs['max_timesteps']:
            print('Maximal timesteps reached')

        # End if at minimum `action_std`
        if policy.action_std <= policy.action_std_min:
            print(f'Ending early on episode {episode} and timestep {timestep}')
            break

        # Activate next stage or decay
        stage += 1
        # CLI
        print(f'Advancing training to stage {stage}')
        if stage < len(stages_kwargs['env']):
            # Activate next stage
            env.set_rewards(stages_kwargs['env'][stage])
        else:
            # Decay policy randomness
            policy.decay_action_std()
            # CLI
            print(f'Decaying std to {policy.action_std} on episode {episode} and timestep {timestep}')

        # Reset early stopping
        early_stopping.reset()
    timer.log('Early Stopping')

    # Iterate
    episode += 1

# CLI Timer
print()
timer.aggregate('sum')

# Finish wandb
if use_wandb: wandb.finish()
