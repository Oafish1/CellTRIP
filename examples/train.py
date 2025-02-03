# %%
from collections import defaultdict
import os

import numpy as np
import torch
import wandb

import data
import celltrip

# Set params
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_FOLDER = os.path.abspath('')
DATA_FOLDER = os.path.join(BASE_FOLDER, '../data/')
MODEL_FOLDER = os.path.join(BASE_FOLDER, 'models/')
if not os.path.isdir(MODEL_FOLDER): os.makedirs(MODEL_FOLDER)

# %% [markdown]
# - TODO
#   - Add module handling defaults
#   - Add train and validation to imputation
#   - Add folder and wandb arguments
#   - Record which partition is used in WandB
#   - Check that rewards are normalized after (?) advantage
#   - Rerun imputation applications with scaled=False (?) and total_statistics=True
#   - Make sure that `scaled=True` on `euclidean_dist` for env partitions is justified
#   - Add multithreading to forward and distributed to backward (ray)
#   - Fix reconstruction speed of memories, will result in 10x training speedup (likely the cause for low GPU utilization)
# 
# - LINKS
#   - [Original paper (pg 24)](https://arxiv.org/pdf/1909.07528.pdf)
#   - [Original blog](https://openai.com/research/emergent-tool-use)
#   - [Gym](https://gymnasium.farama.org/)
#   - [Slides](https://glouppe.github.io/info8004-advanced-machine-learning/pdf/pleroy-hide-and-seek.pdf)
#   - [PPO implementation](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L38)
#   - [Residual SA](https://github.com/openai/multi-agent-emergence-environments/blob/bafaf1e11e6398624116761f91ae7c93b136f395/ma_policy/layers.py#L89)

# %% [markdown]
# ## Arguments

# %%
# Arguments
import argparse
parser = argparse.ArgumentParser(description='Train CellTRIP model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Cast arguments
str_or_int = lambda x: int(x) if x.isdecimal() else x
int_or_none = lambda x: int(x) if x.lower() != 'none' else None

# Important parameters
group = parser.add_argument_group('General')
group.add_argument('--seed', default=42, type=int, help='**Seed for random calls during training')
group.add_argument('--gpu', default='0', type=str, help='**GPU to use for computation')

# Data parameters
group = parser.add_argument_group('Data')
group.add_argument('--dataset', type=str, required=True, help='Dataset to use')
group.add_argument('--no_standardize', action='store_true', help='Don\'t standardize data')
group.add_argument('--top_variant', type=int_or_none, nargs='*', help='Top variant features to filter for each modality')
group.add_argument('--pca_dim', default=[512, 512], type=int_or_none, nargs='*', help='PCA features to generate for each modality')
group.add_argument('--num_nodes', type=int, nargs='*', help='Nodes to sample from data for each episode')

# Environment parameters
group = parser.add_argument_group('Environment')
# TODO: Add more from class
group.add_argument('--dim', default=16, type=int, help='CellTRIP output latent space dimension')
group.add_argument('--reward_distance_target', type=int, nargs='*', help='Target modalities for imputation, leave empty for imputation')

# Environment reward weights
group.add_argument('--env_stages', default=[0b00001, 0b10001, 0b10111, 0b01111], type=int, nargs='*', help=(
    'Environment stages for training, as input, integers are expected with each binary '
    'digit acting as a flag for `penalty_bound`, `penalty_velocity`, `penalty_action`, `reward_distance`, `reward_origin`, respectively'))

# Policy parameters
group = parser.add_argument_group('Policy')
group.add_argument('--max_nodes', type=int, help='**Max number of nodes to include in a single computation (i.e. 100 = 1 self node, 99 neighbor nodes)')
group.add_argument('--sample_strategy', choices=('random', 'proximity', 'random-proximity'), default='random-proximity', type=str, help='Neighbor sampling strategy to use if `max_nodes` is fewer than in state')
group.add_argument('--reproducible_strategy', default='hash', type=str_or_int, help='Method to enforce reproducible sampling between forward and backward, may be `hash` or int')
# Backpropagation
group.add_argument('--update_maxbatch', type=int, help='**Total number of memories to sample from during backprop')
group.add_argument('--update_batch', default=int(1e4), type=int, help='**Number of memories to sample from during each backprop epoch')
group.add_argument('--update_minibatch', default=int(1e4), type=int, help='**Max memories to backprop at a time')
group.add_argument('--update_load_level', default='minibatch', choices=('maxbatch', 'batch', 'minibatch'), help='**What stage to reconstruct memories from compressed form')
group.add_argument('--update_cast_level', default='minibatch', choices=('maxbatch', 'batch', 'minibatch'), type=str, help='**What stage to cast to GPU memory')
# Internal arguments
group.add_argument('--feature_embed_dim', default=32, type=int, help='Dimension of modal embedding')
group.add_argument('--embed_dim', default=64, type=int, help='Internal dimension of state representation')
# Training Arguments
group.add_argument('--action_std_init', default=.6, type=float, help='Initial policy randomness, in std')
group.add_argument('--action_std_decay', default=.05, type=float, help='Policy randomness decrease per stage iteration')
group.add_argument('--action_std_min', default=.15, type=float, help='Final policy randomness')
group.add_argument('--memory_prune', default=100, type=int, help='How many memories to prune from the end of the data')

# Training parameters
group = parser.add_argument_group('Training')
group.add_argument('--max_ep_timesteps', default=int(1e3), type=int, help='Number of timesteps per episode')
group.add_argument('--max_timesteps', default=int(5e6), type=int, help='Absolute max timesteps')
group.add_argument('--update_timesteps', default=int(5e3), type=int, help='Number of timesteps per policy update')
group.add_argument('--max_batch', default=None, type=int, help='**Max number of nodes to calculate actions for at a time')
group.add_argument('--no_episode_random_samples', action='store_true', help='Don\'t refresh episode each epoch')
group.add_argument('--episode_partitioning_feature', type=int, help='Type feature to partition by for episode random samples')
group.add_argument('--use_wandb', action='store_true', help='**Record performance to wandb')

# Early stopping parameters
group = parser.add_argument_group('Early Stopping')
# TODO: Add disable option
group.add_argument('--buffer', default=6, type=int, help='Leniency for early stopping criterion, in updates')
group.add_argument('--window_size', default=3, type=int, help='Window size for early stopping criterion evaluation, in updates')

# Notebook defaults and script handling
if not celltrip.utilities.is_notebook():
    args = parser.parse_args()
else:
    args = parser.parse_args('--dataset ExSeq --max_nodes 100 --pca_dim None'.split(' '))

# %%
# Parse groups
# https://stackoverflow.com/a/46929320
# TODO: Remove `None` entries
arg_groups = {}
for group in parser._action_groups:
    group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    arg_groups[group.title] = argparse.Namespace(**group_dict)
# Convert to dict
arg_groups = {k1: vars(v1) for k1, v1 in arg_groups.items()}
# Invert NOs
arg_groups = {k1: {k2[3:] if k2.startswith('no_') else k2: not v2 if k2.startswith('no_') else v2 for k2, v2 in v1.items()} for k1, v1 in arg_groups.items()}
# Check list args for None values
arg_groups = {k1: {k2: v2[0] if isinstance(v2, list) and len(v2) == 1 and v2[0] is None else v2 for k2, v2 in v1.items()} for k1, v1 in arg_groups.items()}
# Scale early stopping parameters
for k in ('buffer', 'window_size'):
    arg_groups['Early Stopping'][k] *= int(arg_groups['Training']['update_timesteps'] / arg_groups['Training']['max_ep_timesteps'])
# Default dimension parameters
arg_groups['Policy']['positional_dim'] = 2*arg_groups['Environment']['dim']
arg_groups['Policy']['output_dim'] = arg_groups['Environment']['dim']

# Unencode env stages
env_stages_encoded = arg_groups['Environment'].pop('env_stages')
stage_order = ('penalty_bound', 'penalty_velocity', 'penalty_action', 'reward_distance', 'reward_origin')
arg_groups['Stages'] = []
for num in env_stages_encoded:
    arg_groups['Stages'].append({})
    for stage in stage_order:
        arg_groups['Stages'][-1][stage] = num & 1
        num = num >> 1

# Set env vars
os.environ['CUDA_VISIBLE_DEVICES']=arg_groups['General']['gpu']

# %% [markdown]
# ## Load Data

# %%
# Reproducibility
# torch.use_deterministic_algorithms(True)
torch.manual_seed(arg_groups['General']['seed'])
if torch.cuda.is_available(): torch.cuda.manual_seed(arg_groups['General']['seed'])
np.random.seed(arg_groups['General']['seed'])

# Load data
modalities, types, features = data.load_data(arg_groups['Data']['dataset'], DATA_FOLDER)

# Filter data (TemporalBrain)
# mask = [(t.startswith('Adol') or t.startswith('Inf')) for t in types[0][:, 1]]
# modalities, types = [m[mask] for m in modalities], [t[mask] for t in types]

# Preprocess data
ppc = celltrip.utilities.Preprocessing(**arg_groups['Data'], device=DEVICE)
processed_modalities, features = ppc.fit_transform(modalities, features)
modalities = processed_modalities

# Fixed samples
if not arg_groups['Training']['episode_random_samples']:
    processed_modalities, keys = ppc.subsample(processed_modalities, return_idx=True)
    processed_modalities = ppc.cast(processed_modalities)
    modalities = processed_modalities

# CLI
else:
    if arg_groups['Training']['episode_partitioning_feature'] is not None:
        names, counts = np.unique(types[0][:, arg_groups['Training']['episode_partitioning_feature']], return_counts=True)
        print('Episode groups: ' + ', '.join([f'{n} ({c})' for n, c in zip(names, counts)]))

# %% [markdown]
# ## Train Policy

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

# Initialize classes
env = celltrip.environments.trajectory(*modalities, **arg_groups['Environment'], **arg_groups['Stages'][0], device=DEVICE)  # Set to first stage
arg_groups['Policy']['modal_dims'] = [m.shape[1] for m in env.get_return_modalities()]
policy = celltrip.models.PPO(**arg_groups['Policy'], device=DEVICE).train()
early_stopping = celltrip.utilities.EarlyStopping(**arg_groups['Early Stopping'])

# Initialize wandb
if arg_groups['Training']['use_wandb']: wandb.init(
    project='CellTRIP',
    config={
        **{'note/'+k:v for k, v in arg_groups["General"].items()},
        **{'data/'+k:v for k, v in arg_groups["Data"].items()},
        **{'env/'+k:v for k, v in arg_groups["Environment"].items()},
        **{'stages/'+k:v for k, v in arg_groups["Stages"].items()},
        **{'policy/'+k:v for k, v in arg_groups["Policy"].items()},
        **{'train/'+k:v for k, v in arg_groups["Training"].items()},
        **{'es/'+k:v for k, v in arg_groups["Early Stopping"].items()},
    },
)

# Initialize logging vars
torch.cuda.reset_peak_memory_stats()
timer = celltrip.utilities.time_logger(discard_first_sample=True)
timestep = 0; episode = 1; stage = 0

# CLI
print('Beginning training')

# Simulation loop
while timestep < arg_groups['Training']['max_timesteps']:
    # Sample new data
    if arg_groups['Training']['episode_random_samples']:
        modalities, keys = ppc.subsample(
            processed_modalities,
            # NOTE: Partitioning currently only supports aligned modalities
            partition=types[0][:, arg_groups['Training']['episode_partitioning_feature']] if arg_groups['Training']['episode_partitioning_feature'] is not None else None,
            return_idx=True)
        modalities = ppc.cast(modalities)
        env.set_modalities(modalities)

    # Reset environment
    env.reset()
    timer.log('Reset Environment')

    # Start episode
    ep_timestep = 0; ep_reward = 0; ep_itemized_reward = defaultdict(lambda: 0)
    while ep_timestep < arg_groups['Training']['max_ep_timesteps']:
        with torch.no_grad():
            # Get current state
            state = env.get_state(include_modalities=True)
            timer.log('Environment Setup')

            # Get actions from policy
            actions = policy.act_macro(
                state,
                keys=keys,
                max_batch=arg_groups['Training']['max_batch'],
            ).detach()
            timer.log('Calculate Actions')

            # Step environment and get reward
            rewards, finished, itemized_rewards = env.step(actions, return_itemized_rewards=True)
            finished = finished or (ep_timestep == arg_groups['Training']['max_ep_timesteps']-1)  # Maybe move logic inside env?
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
        if timestep % arg_groups['Training']['update_timesteps'] == 0:
            # assert False
            print(f'Updating model with average reward {np.mean(sum(policy.memory.storage["rewards"], []))} on episode {episode} and timestep {timestep}', end='')
            policy.update()
            print(f' ({torch.cuda.max_memory_allocated() / 1024**3:.2f} GB CUDA)')
            torch.cuda.reset_peak_memory_stats()
            timer.log('Update Policy')

        # Escape if finished
        if finished: break

    # Upload stats
    ep_reward = (ep_reward / ep_timestep).item()
    update = int(timestep / arg_groups['Training']['update_timesteps'])
    if arg_groups['Training']['use_wandb']:
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
    if early_stopping(ep_reward) or timestep >= arg_groups['Training']['max_timesteps']:
        # Save model
        wgt_file = os.path.join(MODEL_FOLDER, f'policy_{stage:02}.wgt')
        torch.save(policy.state_dict(), wgt_file)  # Save just weights
        if arg_groups['Training']['use_wandb']: wandb.save(wgt_file)
        # mdl_file = os.path.join(MODEL_FOLDER, f'policy_{stage:02}.mdl')
        # torch.save(policy, mdl_file)  # Save whole model
        # if train_kwargs['use_wandb']: wandb.save(mdl_file)

        # End if maximum timesteps reached
        if timestep >= arg_groups['Training']['max_timesteps']:
            print('Maximal timesteps reached')

        # End if at minimum `action_std`
        if policy.action_std - policy.action_std_min <= 1e-3:
            print(f'Ending early on episode {episode} and timestep {timestep}')
            break

        # Activate next stage or decay
        stage += 1
        print(f'Advancing training to stage {stage}')
        if stage < len(arg_groups['Stages']):
            # Activate next stage
            env.set_rewards(arg_groups['Stages'][stage])
        else:
            # Decay policy randomness
            policy.decay_action_std()
            # CLI
            print(f'Decaying std to {policy.action_std} on episode {episode} and timestep {timestep}')
        # stage += 1  # Stage var is one ahead of index

        # Reset early stopping
        early_stopping.reset()
    timer.log('Early Stopping')

    # Iterate
    episode += 1

# CLI Timer
print()
timer.aggregate('sum')

# Finish wandb
if arg_groups['Training']['use_wandb']: wandb.finish()


