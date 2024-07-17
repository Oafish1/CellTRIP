# %%
## To Check
# Check that rewards are normalized after (?) advantage

## High Priority Training Changes
# Make backward (MAX_NODES, MAX_BATCH) batching work
# Add multithreading to forward and distributed to backward
# Add compatibility for env being on CPU, check for timing changes

## Backburner Priority Training Changes
# Add compatibility for cells with missing modalities (add mask to distance reward)
# Try imitation learning to better learn CT trajectories
# Add parallel envs of different sizes, with different data to help generality
# Fix off-center positioning in large environments
# Revise distance reward - Maybe add cell attraction (all should be close to each other) and repulsion (repulsion based on distance in modality)
# Revise velocity and action penalties to encourage early cell-type separation (i.e. sqrt of vec length or similar)

## Bookkeeping and QOL
# Save every time early stopping occurs
# Hook up sweeps API for wandb

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
import sys
arg1 = int(sys.argv[1])

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
dataset_name = 'scNMT'

if dataset_name == 'scNMT':
    dataset_dir = os.path.join(DATA_FOLDER, 'UnionCom/scNMT')
    M1 = pd.read_csv(os.path.join(dataset_dir, 'Paccessibility_300.txt'), delimiter=' ', header=None).to_numpy()
    M2 = pd.read_csv(os.path.join(dataset_dir, 'Pmethylation_300.txt'), delimiter=' ', header=None).to_numpy()
    M3 = pd.read_csv(os.path.join(dataset_dir, 'RNA_300.txt'), delimiter=' ', header=None).to_numpy()
    T1 = pd.read_csv(os.path.join(dataset_dir, 'type1.txt'), delimiter=' ', header=None).to_numpy().flatten()
    T2 = pd.read_csv(os.path.join(dataset_dir, 'type2.txt'), delimiter=' ', header=None).to_numpy().flatten()
    T3 = pd.read_csv(os.path.join(dataset_dir, 'type3.txt'), delimiter=' ', header=None).to_numpy().flatten()

elif dataset_name == 'BrainChromatin':
    nrows = None  # 2_000
    M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_rna_counts.tsv'), delimiter='\t', nrows=nrows).transpose()  # 4.6 Gb in memory
    M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'brainchromatin/multiome_atac_gene_activities.tsv'), delimiter='\t', nrows=nrows).transpose()  # 2.6 Gb in memory
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
num_nodes = 100  # M1.shape[0]
modalities = [[M1, M2, M3][arg1]]
types = [[T1, T2, T3][arg1]]

# Modify data
modalities = inept.utilities.normalize(*modalities, keep_array=True)  # Normalize
# modalities = inept.utilities.pca_features(*modalities, num_features=(512, 512), keep_array=True)  # PCA features (2 min for 8k x 35+k)
subsample = inept.utilities.subsample_nodes(*modalities, *types, num_nodes=num_nodes, keep_array=True)  # Subsample nodes
modalities, types = subsample[:len(modalities)], subsample[len(modalities):]
# modalities = inept.utilities.subsample_features(*modalities, num_features=(16, 16), keep_array=True)  # Subsample features

# Cast types
modalities = [torch.tensor(Mx, dtype=torch.float32, device=DEVICE) for Mx in modalities]

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
    'pos_bound': 10,
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
max_ep_timesteps = 1e3
update_timesteps = 5 * max_ep_timesteps
max_timesteps = 1e3 * update_timesteps
MAX_BATCH = min( 500, data_kwargs['num_nodes'] )  # NOTE: value should be similar to update_minibatch, if a bit larger
MAX_NODES = min( 50, data_kwargs['num_nodes'] )  # Larger means smaller minibatches but a fuller picture for each agent
MAX_BATCH = MAX_NODES = None  # TODO: Currently values other than `None` do not work with update
train_kwargs = {
    'max_ep_timesteps': max_ep_timesteps,
    'max_timesteps': max_timesteps,
    'update_timesteps': update_timesteps,
    'max_batch': MAX_BATCH,  # Max number of nodes to calculate actions for at a time
    'max_nodes': MAX_NODES,  # Max number of nodes to use as neighbors in action calculation
}

# Policy parameters
# num_train_nodes = data_kwargs['num_nodes'] if train_kwargs['max_nodes'] is None else min(data_kwargs['num_nodes'], train_kwargs['max_nodes'])
# GPU_MEMORY = 6; CPU_MEMORY = 16  # Optimized for 6Gb VRAM and 16Gb RAM
# MAX_GPU_RUN_SAMPLES = int( .8 * (GPU_MEMORY / 6) * 1e4 * (2000 / sum(M.shape[1] for M in modalities)) * (20 / num_train_nodes) )
# GPU_STORE_SAMPLES = int( 2 * MAX_GPU_RUN_SAMPLES )  # 3
# MAX_CPU_SAMPLES = int( (CPU_MEMORY / GPU_MEMORY) * MAX_GPU_RUN_SAMPLES )
# IDEAL_BATCH_SIZE = int( max_ep_timesteps )
update_maxbatch = None  # `MAX_CPU_SAMPLES`, `None` takes slightly longer but is more reliable
update_batch = int(1e3)  # Same or larger size as `update_maxbatch` skips GPU cast step inside epoch loop
update_minibatch = int(1e3)
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
    'memory_gamma': .95,
    'memory_prune': 100,
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    'lr_gamma': 1,
    'update_maxbatch': update_maxbatch,  # Batch to load into RAM
    'update_batch': update_batch,  # Batch to load into VRAM
    'update_minibatch': update_minibatch,  # Batch to compute
    'device': DEVICE,
    # Layer arguments
    'embed_dim': 64,
    'feature_embed_dim': 32,
    'rs_nset': 1e5,  # Inversely proportional to influence of individual reward on moving statistics
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
# Run script and put following above function to profile
#    from memory_profiler import profile
#    @profile
# Use cProfiler to profile timing:
#    python -m cProfile -s time -o profile.prof train.py
#    snakeviz profile.prof
use_wandb = True

# Initialize classes
env = inept.environments.trajectory(*modalities, **env_kwargs, **stages_kwargs['env'][0], device=DEVICE)  # Set to first stage
policy = inept.models.PPO(**policy_kwargs).train()
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
num_train_nodes = data_kwargs['num_nodes'] if train_kwargs['max_nodes'] is None else min(data_kwargs['num_nodes'], train_kwargs['max_nodes'])
print(
    f'Training using {num_train_nodes} nodes out of a'
    f' total {data_kwargs["num_nodes"]} with batches of'
    f' size {train_kwargs["max_batch"]}.'
)
update_maxbatch_print = (
    policy_kwargs["update_maxbatch"]
    if policy_kwargs["update_maxbatch"] is not None else 
    'all'
)
print(
    f'Training on {update_maxbatch_print} states'
    f' with batches of size {policy_kwargs["update_batch"]}'
    f' and minibatches of size {policy_kwargs["update_minibatch"]}'
    f' from {int(train_kwargs["update_timesteps"] * data_kwargs["num_nodes"])} total.')

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
            actions = policy.act_macro(
                state,
                keys=list(range(num_nodes)),
                max_batch=train_kwargs['max_batch'],
                max_nodes=train_kwargs['max_nodes'],
            ).detach()
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


