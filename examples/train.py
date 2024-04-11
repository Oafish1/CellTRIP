# %%
# TODO
# Try with larger environment, so nodes aren't pushed to the side
# Try with 100s of nodes
# Add parallel envs of different sizes, with different data to help generality
# Save checkpoint models

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
import sys
arg1 = int(sys.argv[1])

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
# torch.backends.cudnn.deterministic=True

note_kwargs = {'seed': seed}

# %% [markdown]
# ### Create Environment

# %%
# Load MMD-MA data
M1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped1.txt'), delimiter='\t', header=None).to_numpy()
M2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_mapped2.txt'), delimiter='\t', header=None).to_numpy()
T1 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type1.txt'), delimiter='\t', header=None).to_numpy()
T2 = pd.read_csv(os.path.join(DATA_FOLDER, 'UnionCom/MMD/s1_type2.txt'), delimiter='\t', header=None).to_numpy()

# Data
num_nodes = arg1
M1 = torch.rand((num_nodes, 8), device=DEVICE)
M2 = torch.rand((num_nodes, 16), device=DEVICE)
modalities = (M1, M2)

# Environment
# x, y, vx, vy
num_dims = 2
env = inept.environments.trajectory(*modalities, dim=num_dims, reward_type='euclidean', device=DEVICE)

# Data parameters
env_kwargs = {
    'dataset': 'Random',
    'num_nodes': num_nodes,
    'num_dims': num_dims,
}

# %% [markdown]
# ### Train Policy

# %%
# Policy parameters
input_dims = 2*num_dims+sum([m.shape[1] for m in modalities])
update_minibatch = int( 4e4 * (10 / num_nodes) )
update_max_batch = 1 * update_minibatch  # Only run one minibatch, empirically the benefit is minimal compared to time loss
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
max_ep_timesteps = 2e2
max_timesteps = 1e6
update_timesteps = 20 * max_ep_timesteps
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
use_wandb = True
if use_wandb: wandb.init(
    project='INEPT',
    config={
        **{'note/'+k:v for k, v in note_kwargs.items()},
        **{'data/'+k:v for k, v in env_kwargs.items()},
        **{'policy/'+k:v for k, v in policy_kwargs.items()},
        **{'train/'+k:v for k, v in train_kwargs.items()},
        **{'es/'+k:v for k, v in es_kwargs.items()},
    },
)

# Initialize logging vars
torch.cuda.reset_peak_memory_stats()
timer = inept.utilities.time_logger(discard_first_sample=True)
timestep = 0; episode = 1; recording = defaultdict(lambda: []); stats = defaultdict(lambda: [])

# CLI
print('Beginning training')
print(f'Subsampling {update_max_batch} states with minibatches of size {update_minibatch} from {int(update_timesteps * num_nodes)} total.')

# Simulation loop
while timestep < max_timesteps:
    # Reset environment
    env.reset()
    timer.log('Reset Environment')

    # Start episode
    ep_timestep = 0; ep_reward = 0
    while ep_timestep < max_ep_timesteps:
        with torch.no_grad():
            # Get current state
            state = env.get_state(include_modalities=True)
            recording['states'].append(state.cpu().numpy())

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
            rewards, finished = env.step(actions)
            timer.log('Step Environment')

            # Record rewards
            for key in range(num_nodes):
                policy.memory.rewards.append(rewards[key].item())  # Could just add lists
                policy.memory.is_terminals.append(finished)
            ep_reward = ep_reward + rewards.cpu().sum()
            recording['rewards'].append(rewards.cpu().numpy())
            timer.log('Record Rewards')

        # Iterate
        timestep += 1
        ep_timestep += 1

        # Update model
        if timestep % update_timesteps == 0:
            print(f'Updating model with average reward {np.mean(policy.memory.rewards)} on episode {episode} and timestep {timestep}', end='')
            policy.update()
            print(f' ({torch.cuda.max_memory_allocated() / 1024**3:.2f} GB)')
            torch.cuda.reset_peak_memory_stats()
            timer.log('Update Policy')

        # Escape if finished
        if finished: break

    # Record stats
    ep_reward = (ep_reward / (num_nodes * ep_timestep)).item()
    stats['episode'].append(episode)
    stats['end_timestep'].append(timestep)
    stats['average_reward'].append(ep_reward)
    stats['action_std'].append(policy.action_std)
    stats['update'].append(int(timestep / update_timesteps))

    if use_wandb:
        wandb.log({
            'episode': episode,
            'update': int(timestep / update_timesteps),
            'end_timestep': timestep,
            'average_reward': ep_reward,
            'action_std': policy.action_std,
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
