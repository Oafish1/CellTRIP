from collections import defaultdict
from itertools import product
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
import sklearn.metrics
import torch
from tqdm import tqdm
import wandb

import data
import inept

# Get args
import sys
run_id = sys.argv[1]
key = sys.argv[2]
num_nodes = int(sys.argv[3])

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
#   - Add more accuracy metrics
#   - Perturbation analysis with inverse transform
#   - Add 2D functionality
#   - Add optional UMAP
# 
# - LOW PRIORITY
#   - Switch to `mayavi` instead of mpl to have true 3d and proper layering

# %% [markdown]
# ### Load All Classes

# %%
# Parameters
stage_override = None  # Manually override policy stage selection
seed_override = None  # 43

# Load run
api = wandb.Api()
run = api.run(f'oafish/INEPT/{run_id}')
config = defaultdict(lambda: {})
for k, v in run.config.items():
    dict_name, keyr = k.split('/')
    config[dict_name][keyr] = v
config = dict(config)

# Reproducibility
seed = seed_override if seed_override is not None else config['note']['seed']
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Load data
modalities, types, features = data.load_data(config['data']['dataset'], DATA_FOLDER)
data_dict = config['data']
data_dict = inept.utilities.overwrite_dict(data_dict, {'standardize': True})  # Old model compatibility
if num_nodes is not None: data_dict = inept.utilities.overwrite_dict(data_dict, {'num_nodes': num_nodes})
ppc = inept.utilities.Preprocessing(**data_dict, device=DEVICE)
modalities = ppc.fit_transform(modalities)
modalities, types = ppc.subsample(modalities, types)
modalities = ppc.cast(modalities)

# Load env
env = inept.environments.trajectory(*modalities, **config['env'], device=DEVICE)

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
print(f'Policy found at stage {latest_mdl[0]}')

# Load file
load_type = 'wgt'
if load_type == 'mdl':
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_mdl[1].download(tmpdir, replace=True)
        policy = torch.load(os.path.join(tmpdir, latest_mdl[1].name))
elif load_type == 'wgt':
    # Mainly used in the case of old argument names, also more secure
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_wgt[1].download(tmpdir, replace=True)
        # config_to_use = config['policy']
        config_to_use = inept.utilities.overwrite_dict(config['policy'], {'positional_dim': 6, 'modal_dims': [76]})  # Old model compatibility
        policy = inept.models.PPO(**config_to_use)
        incompatible_keys = policy.load_state_dict(torch.load(os.path.join(tmpdir, latest_wgt[1].name), weights_only=True))
policy = policy.to(DEVICE).eval()
policy.actor.set_action_std(1e-7)

# %%
# TODO: Standardize implementation
times = types[0]  # Temporary time annotation, will change per-dataset

# %% [markdown]
# ### Generate Runs

# %%
# Choose key
# TODO: Calculate both, plot one (?)

# %%
# Initialize memories
memories = {}

# Default present function
def get_present_default(
    *args,
    timestep,
    **kwargs,
):
    return torch.ones(modalities[0].shape[0], dtype=bool), timestep+1 >= config['train']['max_ep_timesteps']

# %% [markdown]
# ##### Integration

# %%
# Deployment list
deployment = [None]
# Reverse alphabetical (ExSeq, MERFISH, smFISH, ISS, MouseVisual)
type_order = np.unique(types[0])[::-1]
deployment_general = {
    'labels': list(type_order),
    'delay': 50*np.arange(len(type_order)),
    'rates': [1] + [.015]*(len(type_order)-1),
    'origins': [None] + list(type_order[:-1])}
deployment += [deployment_general]

# %%
# Choose Deployment
deployment = deployment[1]

# Functions
# Takes in combination of variables, outputs present, end
def get_present_deployment(
    *args,
    env,
    timestep,
    present,
    labels,
    **kwargs,
):
    # Copy status
    present = present.clone()
    state = env.get_state().clone()

    # Iterate over each label
    for label, delay, rate, origin in zip(*deployment.values()):
        # If delay has been reached
        if timestep >= delay:
            # Look at each node
            for i in range(len(present)):
                # If label matches and not already present
                if labels[i] == label and not present[i]:
                    # Roll for appearance
                    if np.random.rand() < rate:
                        # Mark as present and set origin
                        if origin is not None:
                            state[i] = state[np.random.choice(np.argwhere((labels==origin)*present.cpu().numpy()).flatten())]
                        present[i] = True

    # Return
    env.set_state(state)
    return present, timestep+1 >= config['train']['max_ep_timesteps']

# %% [markdown]
# ##### Trajectory

# %%
# Stage order list
temporal = [None]
# Reverse alphabetical (ExSeq, MERFISH, smFISH, ISS, MouseVisual)
temporal_general = {'stages': [[l] for l in np.unique(types[0])[::-1]]}
temporal += [temporal_general]

# %%
# Choose stage order
temporal = temporal[1]

# Functions
current_stage = None  # TODO: Move into class
stage_start = 0
max_stage_len = 500

def get_present_temporal(
    *args,
    timestep,
    env,
    times,  # np.array
    present,
    vel_threshold=3e-2,
    **kwargs,
):
    # Clone data
    present = present.clone()
    state = env.get_state().clone()

    # Defaults
    global current_stage, stage_start
    if timestep == 0:
        current_stage = 0
        stage_start = 0

    # Initiate change if vel is low
    if present.sum() > 0: vel_threshold_met = state[present, env.dim:].square().sum(dim=-1).sqrt().max(dim=-1).values < vel_threshold
    else: vel_threshold_met = False

    update = vel_threshold_met or timestep - stage_start >= max_stage_len
    if update:
        # Make change to next stage
        current_stage += 1
        stage_start = timestep
        if current_stage >= len(temporal['stages']): return present, True
    
    # Update present if needed
    if update or timestep == 0:
        present = torch.tensor(np.isin(times, temporal['stages'][current_stage]))
    
    return present, False

# %% [markdown]
# ##### Main Run

# %%
# Choose key
get_present_dict = {
    'discovery': get_present_deployment if deployment is not None else get_present_default,
    'temporal': get_present_temporal if temporal is not None else get_present_default,
}
get_present_func = get_present_dict[key]

# Initialize
env.reset(); memories[key] = defaultdict(lambda: [])

# Modify
present = torch.zeros(modalities[0].shape[0], dtype=bool, device=DEVICE)
present, _ = get_present_func(
    env=env,
    timestep=0,
    present=present,
    labels=types[0],
    times=times,
    deployment=deployment,
)

# Continue initializing
memories[key]['present'].append(present)
memories[key]['states'].append(env.get_state())
memories[key]['rewards'].append(torch.zeros(modalities[0].shape[0], device=DEVICE))

# Simulate
timestep = 1
while True:
    # CLI
    if timestep % 20 == 0:
        cli_out = f'Timestep: {timestep}'
        if key == 'temporal': cli_out += f' - Stage: {current_stage}'
        print(cli_out, end='\r')

    # Step
    state = env.get_state(include_modalities=True)
    actions = torch.zeros((modalities[0].shape[0], env.dim), device=DEVICE)
    actions[present] = policy.act_macro(
        state[present],
        keys=torch.arange(modalities[0].shape[0], device=DEVICE)[present],
        max_batch=config['train']['max_batch'],
        max_nodes=config['train']['max_nodes'],
    )
    rewards = torch.zeros(modalities[0].shape[0])
    # TODO: Currently, rewards factor in non-present nodes
    rewards, _, _ = env.step(actions, return_itemized_rewards=True)
    new_state = env.get_state()
    new_state[~present] = state[~present, :2*env.dim]  # Don't move un-spawned nodes
    env.set_state(new_state)

    # Record
    memories[key]['present'].append(present)
    memories[key]['states'].append(env.get_state())
    memories[key]['rewards'].append(rewards)

    # Modify
    present, end = get_present_func(
        env=env,
        timestep=timestep, 
        present=present, 
        labels=types[0],
        times=times,
        deployment=deployment,
    )

    # End
    if end: break
    timestep += 1

# Stack
memories[key]['present'] = torch.stack(memories[key]['present'])
memories[key]['states'] = torch.stack(memories[key]['states'])
memories[key]['rewards'] = torch.stack(memories[key]['rewards'])
memories[key] = dict(memories[key])

# CLI
print()

# %% [markdown]
# ### Plot Memories

# %% [markdown]
# ##### Integration

# %%
# Prepare data
skip = 10
present = memories[key]['present'].cpu()[::skip]
states = memories[key]['states'].cpu()[::skip]
rewards = memories[key]['rewards'].cpu()[::skip]
env.set_modalities(modalities)
env.reset()
env.get_distance_match()
modal_dist = env.dist

# Parameters
interval = 1e3*env.delta/3  # Time between frames (3x speedup)
min_max_vel = 0 if key == 'temporal' else 1e-2  # Stop at first frame all vels are below target. 0 for full play
frame_override = None  # Manually enter number of frames to draw
num_lines = 25  # Number of attraction and repulsion lines
rotations_per_second = .1  # Camera azimuthal rotations per second

# Create plot
figsize = (17, 10)
fig = plt.figure(figsize=figsize)
# grid = mpl_grid.GridSpec(1, 2, width_ratios=(2, 1))
# ax1 = fig.add_subplot(grid[0], projection='3d')
# ax2 = fig.add_subplot(grid[1])
# fig.tight_layout(pad=2)
ax1 = fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d')
ax2 = fig.add_axes([12 /figsize[0], 1 /figsize[1], 4 /figsize[0], 8 /figsize[1]])

# Initialize nodes
get_node_data = lambda frame: states[frame, :, :3]
nodes = [
    ax1.plot(
        # *get_node_data(0)[types[0]==l].T,
        [], [],
        label=l,
        linestyle='',
        marker='o',
        ms=6,
        zorder=2.3,
    )[0]
    for l in np.unique(types[0])
]

# Initialize velocity arrows
arrow_length_scale = 1
get_arrow_xyz_uvw = lambda frame: (states[frame, :, :3], states[frame, :, env.dim:env.dim+3])
arrows = ax1.quiver(
    [], [], [],
    [], [], [],
    arrow_length_ratio=0,
    length=arrow_length_scale,
    lw=2,
    color='gray',
    alpha=.4,
    zorder=2.2,
)

# Initialize modal lines
# relative_connection_strength = [np.array([(1-dist[j, k].item()/dist.max().item())**2 for j, k in product(*[range(s) for s in dist.shape]) if j < k]) for dist in modal_dist]
get_distance_discrepancy = lambda frame: [np.array([((states[frame, j, :3] - states[frame, k, :3]).square().sum().sqrt() - dist[j, k].cpu()).item() for j, k in product(*[range(s) for s in dist.shape]) if j < k]) for dist in modal_dist]
get_modal_lines_segments = lambda frame, dist: np.array(states[frame, [[j, k] for j, k in product(*[range(s) for s in dist.shape]) if j < k], :3])
clip_dd_to_alpha = lambda dd: np.clip(np.abs(dd), 0, 2) / 2
# Randomly select lines to show
line_indices = [[j, k] for j, k in product(*[range(s) for s in modal_dist[0].shape]) if j < k]
total_lines = int((modal_dist[0].shape[0]**2 - modal_dist[0].shape[0]) / 2)  # Only considers first modality
line_selection = [
    np.random.choice(total_lines, num_lines, replace=False) if num_lines is not None else list(range(total_lines)) for dist in modal_dist
]
modal_lines = [
    mp3d.art3d.Line3DCollection(
        get_modal_lines_segments(0, dist)[line_selection[i]],
        label=f'Modality {i}',
        lw=2,
        zorder=2.1,
    )
    for i, dist in enumerate(modal_dist)
]
for ml in modal_lines: ax1.add_collection(ml)

# Silhouette scoring
if key == 'discovery':
    # TODO: Update from 3 to env.dim
    get_silhouette_samples = lambda frame: sklearn.metrics.silhouette_samples(states[frame, :, :3].cpu(), types[0])
    bars = [ax2.bar(l, 0) for l in np.unique(types[0])]
    ax2.axhline(y=0, color='black')
    ax2.set(ylim=(-1, 1))

# Temporal comparison
elif key == 'temporal':
    def get_temporal_discrepancy(frame, recalculate=True):
        if recalculate: env.set_modalities([m[present[frame], :] for m in modalities])
        env.set_positions(states[frame, present[frame], :env.dim].to(DEVICE))
        return float(env.get_distance_match().mean().detach().cpu())
    temporal_eval_plot = ax2.plot([], [], color='black', marker='o')[0]
    # TODO: Highlight training regions
    num_stages = len(temporal['stages'])  # present.unique(dim=0).shape[0]

    # Styling
    ax2.set_xticks(np.arange(num_stages), temporal['stages'])
    ax2.set_xlim([-.5, num_stages-.5])
    ax2.set_ylim([0, 1e0])
    ax2.set_title('Temporal Discrepancy')
    # ax2.set_yscale('symlog')

# Limits
# TODO: Double-check that the `present` indexing works
ax1.set(
    xlim=(states[present][:, 0].min(), states[present][:, 0].max()),
    ylim=(states[present][:, 1].min(), states[present][:, 1].max()),
    zlim=(states[present][:, 2].min(), states[present][:, 2].max()),
)

# Legends
l1 = ax1.legend(handles=nodes, loc='upper left')
ax1.add_artist(l1)
l2 = ax1.legend(handles=[
    ax1.plot([], [], color='red', label='Repulsion')[0],
    ax1.plot([], [], color='blue', label='Attraction')[0],
], loc='upper right')
ax1.add_artist(l2)
ax2.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

# Styling
ax1.set(xlabel='x', ylabel='y', zlabel='z')
get_angle = lambda frame: (30, (360*rotations_per_second)*(frame*interval/1000)-60, 0)
ax1.view_init(*get_angle(0))

# Update function
def update(frame):
    # Adjust nodes
    for i, l in enumerate(np.unique(types[0])):
        present_labels = present[frame] * torch.tensor(types[0]==l)
        data = get_node_data(frame)[present_labels].T
        nodes[i].set_data(*data[:2])
        nodes[i].set_3d_properties(data[2])

    # Adjust arrows
    xyz_xyz = [[xyz, xyz+arrow_length_scale*uvw] for i, (xyz, uvw) in enumerate(zip(*get_arrow_xyz_uvw(frame))) if present[frame, i]]
    arrows.set_segments(xyz_xyz)

    # Adjust lines
    for i, (dist, ml) in enumerate(zip(modal_dist, modal_lines)):
        ml.set_segments(get_modal_lines_segments(frame, dist)[line_selection[i]])
        distance_discrepancy = get_distance_discrepancy(frame)[i][line_selection[i]]
        color = np.array([(0., 0., 1.) if dd > 0 else (1., 0., 0.) for dd in distance_discrepancy])
        alpha = np.expand_dims(clip_dd_to_alpha(distance_discrepancy), -1)
        for j, line_index in enumerate(line_selection[i]):
            if not present[frame, line_indices[line_index]].all(): alpha[j] = 0.
        ml.set_color(np.concatenate((color, alpha), axis=-1))

    # Barplots
    if key == 'discovery':
        for bar, l in zip(bars, np.unique(types[0])):
            bar[0].set_height(get_silhouette_samples(frame)[types[0]==l].mean())

        # Styling
        ax2.set_title(f'Silhouette Coefficient : {get_silhouette_samples(frame).mean():5.2f}') 

    # Line plots
    elif key == 'temporal':
        # Defaults
        global current_stage  # TODO: Find better solution

        # Calculate discrepancy using env
        recalculate = (frame == 0) or not (present[frame] == present[frame-1]).all()  # Only recalculate dist if needed
        discrepancy = get_temporal_discrepancy(frame, recalculate=recalculate)

        # Adjust plot
        xdata = temporal_eval_plot.get_xdata()
        ydata = temporal_eval_plot.get_ydata()
        if not ((frame == 0 and len(xdata) > 0)):
            if frame == 0: current_stage = 0
            if recalculate:
                xdata = np.append(xdata, current_stage)
                ydata = np.append(ydata, None)
                current_stage += 1  # Technically one ahead
            ydata[-1] = discrepancy
            temporal_eval_plot.set_xdata(xdata)
            temporal_eval_plot.set_ydata(ydata)

    # Styling
    ax1.set_title(f'{skip*frame: 4} : {rewards[frame].mean():5.2f}')  
    ax1.view_init(*get_angle(frame))

    # CLI
    print(f'{frame} / {frames-1}', end='\r')
    if frame == frames-1: print()

# Compile animation
frames = states[..., env.dim:env.dim+3].square().sum(dim=-1).sqrt().max(dim=-1).values < min_max_vel
frames = np.array([(frames[i] or frames[i+1]) if i != len(frames)-1 else frames[i] for i in range(len(frames))])  # Disregard interrupted sections of low movement
frames = np.argwhere(frames)
frames = frames[0, 0].item()+1 if len(frames) > 0 else states.shape[0]
frames = frames if frame_override is None else frame_override
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
ani.save(os.path.join(PLOT_FOLDER, f'{config["data"]["dataset"]}_{key}.{file_type}'), writer=writer, dpi=300)

# CLI
print()
