from collections import defaultdict
import re
import os
import tempfile

import matplotlib as mpl
import matplotlib.collections as mpl_col
import matplotlib.gridspec as mpl_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as mp3d
import numpy as np
import seaborn as sns
import torch
import wandb

import data
import inept

# Set env vars
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# print(torch.cuda.device_count())

# Get args
import sys
run_id_idx = int(sys.argv[1])
# analysis_key_idx = int(sys.argv[2])

# Set params
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_FOLDER = os.path.abspath('')
DATA_FOLDER = os.path.join(BASE_FOLDER, '../data')
PLOT_FOLDER = os.path.join(BASE_FOLDER, '../plots')

# Style
sns.set_context('paper', font_scale=1.25)
sns.set_style('white')
sns.set_palette('husl')

# MPL params
mpl.rcParams['animation.embed_limit'] = 100

# Disable gradients
torch.set_grad_enabled(False);

# %% [markdown]
# - HIGH PRIORITY
#   - Extent `get_present_func` to class with full state-altering capabilities
#   - Add more accuracy metrics
#   - Add 2D functionality
#   - Add optional UMAP
# 
# - LOW PRIORITY
#   - Switch to `mayavi` instead of mpl to have true 3d and proper layering

# %% [markdown]
# ### Load All Classes

# %%
# Parameters
# run_id_idx = 1
run_id = (
    'brf6n6sn',  # TemporalBrain Random 100 Max
    'rypltvk5',  # MMD-MA Random 100 Max
    '32jqyk54',  # MERFISH Random 100 Max
    'c8zsunc9',  # ISS Random 100 Max
    'maofk1f2',  # ExSeq NR
    'f6ajo2am',  # smFish NR
    'vb1x7bae',  # MERFISH NR
    '473vyon2',  # ISS NR
)[run_id_idx]
stage_override = None  # Manually override policy stage selection
num_nodes_override = None
max_batch_override = None
max_nodes_override = None
seed_override = None

# Load run
api = wandb.Api()
run = api.run(f'oafish/INEPT/{run_id}')
config = defaultdict(lambda: {})
for k, v in run.config.items():
    dict_name, key = k.split('/')
    config[dict_name][key] = v
config = dict(config)

# Reproducibility
seed = seed_override if seed_override is not None else config['note']['seed']
# torch.use_deterministic_algorithms(True)
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Load data
modalities, types, features = data.load_data(config['data']['dataset'], DATA_FOLDER)
# config['data'] = inept.utilities.overwrite_dict(config['data'], {'standardize': True})  # Old model compatibility
# config['data'] = inept.utilities.overwrite_dict(config['data'], {'top_variant': [20, 20]})  # Top variant testing
if num_nodes_override is not None: config['data'] = inept.utilities.overwrite_dict(config['data'], {'num_nodes': num_nodes_override})
if max_batch_override is not None: config['train'] = inept.utilities.overwrite_dict(config['train'], {'max_batch': max_batch_override})
ppc = inept.utilities.Preprocessing(**config['data'], device=DEVICE)
modalities, features = ppc.fit_transform(modalities, features)
modalities, types = ppc.subsample(modalities, types)
modalities = ppc.cast(modalities)

# Get latest policy
latest_mdl = [-1, None]  # Pkl
latest_wgt = [-1, None]  # State dict
for file in run.files():
    # Find mdl files
    matches = re.findall(f'^(?:models|trained_models)/policy_(\w+).(mdl|wgt)$', file.name)
    if len(matches) > 0: stage = int(matches[0][0]); ftype = matches[0][1]
    else: continue

    # Record
    latest_known_stage = latest_mdl[0] if ftype == 'mdl' else latest_wgt[0]
    if (stage_override is None and stage > latest_known_stage) or (stage_override is not None and stage == stage_override):
        if ftype == 'mdl': latest_mdl = [stage, file]
        elif ftype == 'wgt': latest_wgt = [stage, file]
print(f'MDL policy found at stage {latest_mdl[0]}')
print(f'WGT policy found at stage {latest_wgt[0]}')

# Load env
env = inept.environments.trajectory(*modalities, **config['env'], **config['stages']['env'][0], device=DEVICE)
for weight_stage in config['stages']['env'][1:latest_mdl[0]+1]:
    env.set_rewards(weight_stage)

# Load file
load_type = 'wgt'
if load_type == 'mdl':
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_mdl[1].download(tmpdir, replace=True)
        policy = torch.load(os.path.join(tmpdir, latest_mdl[1].name))
elif load_type == 'wgt':
    # Mainly used in the case of old argument names, also generally more secure
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_wgt[1].download(tmpdir, replace=True)
        # config['policy'] = inept.utilities.overwrite_dict(config['policy'], {'positional_dim': 6, 'modal_dims': [76]})  # Old model compatibility
        if max_nodes_override is not None: config['policy'] = inept.utilities.overwrite_dict(config['policy'], {'max_nodes': max_nodes_override})
        policy = inept.models.PPO(**config['policy'])
        incompatible_keys = policy.load_state_dict(torch.load(os.path.join(tmpdir, latest_wgt[1].name), weights_only=True))
policy = policy.to(DEVICE).eval()
policy.actor.set_action_std(1e-7)

# %%
# TODO: Standardize implementation
labels = types[0][:, 0]
times = types[0][:, -1]  # Temporary time annotation, will change per-dataset

# %% [markdown]
# ### Generate Runs

# %%
# Choose key
# TODO: Calculate all, plot one (?)
analysis_key_idx = 3
discovery_key = 0  # Auto
temporal_key = 0  # Auto
perturbation_features = [np.random.choice(len(fs), 10, replace=False) for i, fs in enumerate(features) if (i not in env.reward_distance_target) or (len(env.reward_distance_target) == len(modalities))]

analysis_key, state_manager_class = [
    ('integration', inept.utilities.IntegrationStateManager),
    ('discovery', inept.utilities.DiscoveryStateManager),
    ('temporal', inept.utilities.TemporalStateManager),
    ('perturbation', inept.utilities.PerturbationStateManager),
][analysis_key_idx]

# Discovery list
discovery = []
# Reverse alphabetical (ExSeq, MERFISH, smFISH, ISS, MouseVisual)
type_order = np.unique(labels)[::-1]
discovery_general = {
    'labels': list(type_order),
    'delay': 50*np.arange(len(type_order)),
    'rates': [1] + [.015]*(len(type_order)-1),
    'origins': [None] + list(type_order[:-1])}
discovery += [discovery_general]
# Choose Discovery
discovery = discovery[discovery_key]

# Stage order list
temporal = []
# Reverse alphabetical (ExSeq, MERFISH, smFISH, ISS, MouseVisual)
temporal_general = {'stages': [[l] for l in np.unique(labels)[::-1]]}
temporal += [temporal_general]
# Choose stage order
temporal = temporal[temporal_key]

# Perturbation feature names
perturbation_feature_names = [[fnames[pf] for pf in pfs] for pfs, fnames in zip(perturbation_features, features)]

# Initialize memories
memories = {}

# %%
# Choose state manager
state_manager = state_manager_class(
    device=DEVICE,
    discovery=discovery,
    temporal=temporal,
    perturbation_features=perturbation_features,
    modal_targets=env.reward_distance_target,
    num_nodes=modalities[0].shape[0],
    dim=env.dim,
)
get_current_stage = lambda: state_manager.current_stage if state_manager_class in (inept.utilities.TemporalStateManager, inept.utilities.PerturbationStateManager) else -1

# Initialize
env.reset(); memories[analysis_key] = defaultdict(lambda: [])

# Modify
state_vars, end = state_manager(
    # present=present,
    state=env.get_state(),
    modalities=ppc.cast(ppc.inverse_transform(ppc.inverse_cast(env.get_modalities()))),
    labels=labels,
    times=times,
)
present = state_vars['present']
env.set_state(state_vars['state'])
raw_modalities = state_vars['modalities']
env.set_modalities(ppc.cast(ppc.transform(ppc.inverse_cast(raw_modalities))))

# Continue initializing
memories[analysis_key]['present'].append(present)
memories[analysis_key]['states'].append(env.get_state())
memories[analysis_key]['stages'].append(get_current_stage())
memories[analysis_key]['rewards'].append(torch.zeros(modalities[0].shape[0], device=DEVICE))

# Simulate
timestep = 1
while True:
    # CLI
    if timestep % 20 == 0:
        cli_out = f'Timestep: {timestep}'
        if get_current_stage() != -1: cli_out += f' - Stage: {get_current_stage()}'
        print(cli_out, end='\r')

    # Step
    state = env.get_state(include_modalities=True)
    actions = torch.zeros((modalities[0].shape[0], env.dim), device=DEVICE)
    actions[present] = policy.act_macro(
        state[present],
        keys=torch.arange(modalities[0].shape[0], device=DEVICE)[present],
        max_batch=config['train']['max_batch'],
    )
    rewards = torch.zeros(modalities[0].shape[0])
    # TODO: Currently, rewards factor in non-present nodes
    rewards, _, _ = env.step(actions, return_itemized_rewards=True)
    new_state = env.get_state()
    new_state[~present] = state[~present, :2*env.dim]  # Don't move un-spawned nodes
    env.set_state(new_state)

    # Modify
    state_vars, end = state_manager(
        present=present,
        state=env.get_state(),
        modalities=raw_modalities,
        labels=labels,
        times=times,
    )
    present = state_vars['present']
    env.set_state(state_vars['state'])
    # Only modify if changes
    if torch.tensor([(rm != svm).any() for rm, svm in zip(raw_modalities, state_vars['modalities'])]).any():
        raw_modalities = state_vars['modalities']
        env.set_modalities(ppc.cast(ppc.transform(ppc.inverse_cast(raw_modalities))))

    # Record
    memories[analysis_key]['present'].append(present)
    memories[analysis_key]['states'].append(env.get_state())
    memories[analysis_key]['stages'].append(get_current_stage())
    memories[analysis_key]['rewards'].append(rewards)

    # End
    if end: break
    timestep += 1

# Stack
memories[analysis_key]['present'] = torch.stack(memories[analysis_key]['present'])
memories[analysis_key]['states'] = torch.stack(memories[analysis_key]['states'])
memories[analysis_key]['stages'] = torch.tensor(memories[analysis_key]['stages'])
memories[analysis_key]['rewards'] = torch.stack(memories[analysis_key]['rewards'])
memories[analysis_key] = dict(memories[analysis_key])

# CLI
print()

# %%
stages, counts = np.unique(memories[analysis_key]['stages'], return_counts=True)
print('Steps per Stage:')
for s, c in zip(stages, counts):
    print(f'{s}: {c}')

# %%
memories[analysis_key]['rewards'].cpu().mean()  # TODO: Why not the same as WANDB?

# %%
# Perturbation significance analysis
if 'perturbation' in memories:
    # Get last idx for each stage
    stages = memories['perturbation']['stages'].cpu().numpy()
    unique_stages, unique_idx = np.unique(stages[::-1], return_index=True)
    unique_idx = stages.shape[0] - unique_idx - 1
    # unique_stages, unique_idx = unique_stages[::-1], unique_idx[::-1]

    # Record perturbation feature pairs
    perturbation_feature_triples = [(i, f, n) for i, (fs, ns) in enumerate(zip(perturbation_features, perturbation_feature_names)) for f, n in zip(fs, ns)]

    # Compute effect sizes for each
    effect_sizes = []
    for stage, idx in zip(unique_stages, unique_idx):
        # Get state
        state = memories['perturbation']['states'][idx]

        # Record steady state after integration
        if stage == 0:
            steady_state = state
            continue

        # Get perturbed feature
        m_idx, pf, pf_name = perturbation_feature_triples[stage-1]

        # Compute effect size
        effect_size = (state[:, :env.dim] - steady_state[:, :env.dim]).square().sum(dim=-1).sqrt().mean(dim=-1).item()
        effect_sizes.append(effect_size)

    # Print effect sizes
    i = 0
    for j, (pfs, pfns) in enumerate(zip(perturbation_features, perturbation_feature_names)):
        print(f'Modality {j}:')
        for pf, pfn in zip(pfs, pfns):
            print(f'{pfn}:\t{effect_sizes[i]:.02e}')
            i += 1
        print()

# %% [markdown]
# ### Plot Memories

# %%
# Prepare data
skip = 1
present = memories[analysis_key]['present'].cpu()[::skip]
states = memories[analysis_key]['states'].cpu()[::skip]
stages = memories[analysis_key]['stages'].cpu()[::skip]
rewards = memories[analysis_key]['rewards'].cpu()[::skip]
env.set_modalities(modalities)
env.reset()
env.get_distance_match()
modal_dist = env.dist

# Parameters
interval = 1e3*env.delta/3  # Time between frames (3x speedup)
min_max_vel = 1e-2 if analysis_key in ('integration', 'discovery') else -1  # Stop at first frame all vels are below target. 0 for full play
frame_override = None  # Manually enter number of frames to draw
num_lines = 25  # Number of attraction and repulsion lines
rotations_per_second = .1  # Camera azimuthal rotations per second

# Create plot based on key
# NOTE: Standard 1-padding all around and between figures
# NOTE: Left, bottom, width, height
if analysis_key in ('integration', 'discovery'):
    figsize = (15, 10)
    fig = plt.figure(figsize=figsize)
    axs = [
        fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
        fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
        fig.add_axes([10 /figsize[0], 1 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
    ]
    views = [
        inept.utilities.View3D,
        inept.utilities.ViewTemporalScatter,
        inept.utilities.ViewSilhouette,
    ]

elif analysis_key == 'temporal':
    figsize = (15, 10)
    fig = plt.figure(figsize=figsize)
    axs = [
        fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
        fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
        fig.add_axes([10 /figsize[0], 1 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
    ]
    views = [
        inept.utilities.View3D,
        inept.utilities.ViewTemporalScatter,
        inept.utilities.ViewTemporalDiscrepancy,
    ]

elif analysis_key in ('perturbation',):
    figsize = (20, 10)
    fig = plt.figure(figsize=figsize)
    axs = [
        fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
        fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 8 /figsize[0], 3.5 /figsize[1]]),
        fig.add_axes([10 /figsize[0], 1 /figsize[1], 3.5 /figsize[0], 3.5 /figsize[1]]),
        fig.add_axes([14.5 /figsize[0], 1 /figsize[1], 3.5 /figsize[0], 3.5 /figsize[1]]),
    ]
    views = [
        inept.utilities.View3D,
        inept.utilities.ViewPerturbationEffect,
        inept.utilities.ViewTemporalScatter,
        inept.utilities.ViewSilhouette,
    ]

# Initialize views
arguments = {
    # Data
    'present': present,
    'states': states,
    'stages': stages,
    'rewards': rewards,
    'modalities': modalities,
    'labels': labels,
    # Data params
    'dim': env.dim,
    'modal_targets': env.reward_distance_target,
    'temporal_stages': temporal['stages'] if analysis_key == 'temporal' else None,
    'perturbation_features': perturbation_features,
    'perturbation_feature_names': perturbation_feature_names,
    # Arguments
    'interval': interval,
    'skip': skip,
    'seed': 42,
    # Styling
    'ms': 5,  # 3
    'lw': 1,
}
views = [view(**arguments, ax=ax) for view, ax in zip(views, axs)]

# Update function
def update(frame):
    # Update views
    for view in views:
        # print(view)
        view.update(frame)

    # CLI
    print(f'{frame} / {frames-1}', end='\r')
    if frame == frames-1: print()

# Compile animation
frames = states[..., env.dim:env.dim+3].square().sum(dim=-1).sqrt().max(dim=-1).values < min_max_vel
frames = np.array([(frames[i] or frames[i+1]) if i != len(frames)-1 else frames[i] for i in range(len(frames))])  # Disregard interrupted sections of low movement
frames = np.argwhere(frames)
frames = frames[0, 0].item()+1 if len(frames) > 0 else states.shape[0]
frames = frames if frame_override is None else frame_override

# Test individual frames
# for frame in range(frames):
#     update(frame)
#     # print()
#     # print('saving')
#     fig.savefig(os.path.join('temp/plots', f'frame_{frame}.png'), dpi=300)
#     break

# Initialize animation
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=frames,
    interval=interval,
)

# Display animation as it renders
# plt.show()

# Display complete animation
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# Save animation
file_type = 'mp4'
if file_type == 'mp4': writer = animation.FFMpegWriter(fps=int(1e3/interval), extra_args=['-vcodec', 'libx264'], bitrate=8e3)  # Faster
elif file_type == 'gif': writer = animation.FFMpegWriter(fps=int(1e3/interval))  # Slower
ani.save(os.path.join(PLOT_FOLDER, f'{config["data"]["dataset"]}_{analysis_key}.{file_type}'), writer=writer, dpi=300)

# CLI
print()


