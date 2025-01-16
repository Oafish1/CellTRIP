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
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.neighbors
import torch
import wandb

import data
import inept

# Set env vars
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# print(torch.cuda.device_count())

# Get args
import sys
# run_id_idx = int(sys.argv[1])
analysis_key_idx = int(sys.argv[1])
# stage_override = int(sys.argv[1])

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
run_id_idx = 3
run_id = (
    'brf6n6sn',  # TemporalBrain Random 100 Max
    'rypltvk5',  # MMD-MA Random 100 Max (requires `total_statistics`)
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
print(f'Loading run {run_id}...')
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
print(f'Loading dataset {config["data"]["dataset"]}...')
modalities, types, features = data.load_data(config['data']['dataset'], DATA_FOLDER)
# config['data'] = inept.utilities.overwrite_dict(config['data'], {'standardize': True})  # Old model compatibility
# config['data'] = inept.utilities.overwrite_dict(config['data'], {'top_variant': [20, 20]})  # Top variant testing
if num_nodes_override is not None: config['data'] = inept.utilities.overwrite_dict(config['data'], {'num_nodes': num_nodes_override})
if max_batch_override is not None: config['train'] = inept.utilities.overwrite_dict(config['train'], {'max_batch': max_batch_override})
ppc = inept.utilities.Preprocessing(**config['data'], device=DEVICE)
total_statistics = False  # Legacy compatibility
modalities, features = ppc.fit_transform(modalities, features, total_statistics=total_statistics)
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
if load_type == 'mdl' and latest_mdl[0] != -1:
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_mdl[1].download(tmpdir, replace=True)
        policy = torch.load(os.path.join(tmpdir, latest_mdl[1].name))
elif load_type == 'wgt' and latest_wgt[0] != -1:
    # Mainly used in the case of old argument names, also generally more secure
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_wgt[1].download(tmpdir, replace=True)
        # config['policy'] = inept.utilities.overwrite_dict(config['policy'], {'positional_dim': 6, 'modal_dims': [76]})  # Old model compatibility
        if max_nodes_override is not None: config['policy'] = inept.utilities.overwrite_dict(config['policy'], {'max_nodes': max_nodes_override})
        policy = inept.models.PPO(**config['policy'])
        incompatible_keys = policy.load_state_dict(torch.load(os.path.join(tmpdir, latest_wgt[1].name), weights_only=True))
else:
    # Use random model
    policy = inept.models.PPO(**config['policy'])
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
# analysis_key_idx = 0
optimize_memory = True  # Saves memory by shrinking env based on present
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
temporal_general = {'stages': [[l] for l in np.unique(times)[::-1]]}
temporal += [temporal_general]
# Choose stage order
temporal = temporal[temporal_key]

# Perturbation feature names
perturbation_feature_names = [[fnames[pf] for pf in pfs] for pfs, fnames in zip(perturbation_features, features)]

# Initialize memories
memories = {}

# %%
# Profiling
profile = True
if profile: torch.cuda.memory._record_memory_history(max_entries=100000)

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
# TODO: Figure out why -1 on repeat for temporal/perturbation?
get_current_stage = lambda: state_manager.current_stage if state_manager_class in (inept.utilities.TemporalStateManager, inept.utilities.PerturbationStateManager) else -1

# Initialize
env.set_modalities(modalities); env.reset(); memories[analysis_key] = defaultdict(lambda: [])

# Modify
state_vars, end = state_manager(
    # present=present,
    state=env.get_state(),
    # TODO: Somehow replace the inverse_transform with just the original data? For sparse compatibility
    modalities=ppc.cast(ppc.inverse_transform(ppc.inverse_cast(modalities)), device='cpu'),
    labels=labels,
    times=times,
)
present = state_vars['present']
memory_mask = present if optimize_memory else torch.ones_like(present, device=DEVICE)
full_state = state_vars['state']
env.set_state(full_state[memory_mask])
raw_modalities = state_vars['modalities']
env.set_modalities(ppc.cast(ppc.transform(ppc.inverse_cast([m[memory_mask.cpu()] for m in raw_modalities]))))

# Continue initializing
memories[analysis_key]['present'].append(present)
memories[analysis_key]['states'].append(full_state.clone())
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
        state if optimize_memory else state[present],
        keys=torch.arange(modalities[0].shape[0], device=DEVICE)[present],
        max_batch=config['train']['max_batch'],
    )
    rewards = torch.zeros(modalities[0].shape[0], device=DEVICE)
    new_rewards, _, _ = env.step(actions[present] if optimize_memory else actions, return_itemized_rewards=True)
    if optimize_memory: rewards[present] = new_rewards
    else: rewards = new_rewards
    full_state[present] = env.get_state() if optimize_memory else env.get_state()[present]
    if not optimize_memory: env.set_state(full_state)  # Don't move un-spawned nodes

    # Modify
    state_vars, end = state_manager(
        present=present,
        state=full_state,
        modalities=raw_modalities,
        labels=labels,
        times=times,
    )
    present_change = (state_vars['present'] != present).any()
    present = state_vars['present']
    memory_mask = present if optimize_memory else torch.ones_like(present, device=DEVICE)
    full_state = state_vars['state']
    env.set_state(full_state[memory_mask])
    # Only modify if changes
    if (
        torch.tensor([(rm != svm).any() for rm, svm in zip(raw_modalities, state_vars['modalities'])]).any()
        or (optimize_memory and present_change)
    ):
        raw_modalities = state_vars['modalities']
        env.set_modalities(ppc.cast(ppc.transform(ppc.inverse_cast([m[memory_mask.cpu()] for m in raw_modalities]))))

    # Record
    memories[analysis_key]['present'].append(present)
    memories[analysis_key]['states'].append(full_state.clone())
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

# Profiling
if profile:
    torch.cuda.memory._dump_snapshot('snapshot.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)

# CLI
print()

# %%
memories[analysis_key]['rewards'].cpu().mean()

# %%
stages, counts = np.unique(memories[analysis_key]['stages'], return_counts=True)
print('Steps per Stage:')
for s, c in zip(stages, counts):
    print(f'{s}: {c}')

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
# ### Static Visualizations

# %% [markdown]
# ##### Loss Plot

# %%
# Load history from wandb
history = run.history(samples=2000)
history['timestep'] = history['end_timestep']
history['Runtime (h)'] = history['_runtime'] / 60**2

# Plot
fig, ax = plt.subplots(1, 1, figsize=(18, 6), layout='constrained')
def plot_without_zeros(x, y, **kwargs):
    x, y = x[np.argwhere(y != 0).flatten()], y[np.argwhere(y != 0).flatten()]
    ax.plot(x, y, **kwargs)
ax.plot(history['timestep'], history['average_reward'], color='black', lw=3, label='Average Reward')
plot_without_zeros(history['timestep'], history['rewards/bound'], color='red', alpha=.75, lw=2, label='Boundary Penalty')
plot_without_zeros(history['timestep'], history['rewards/velocity'], color='goldenrod', alpha=.75, lw=2, label='Velocity Penalty')
plot_without_zeros(history['timestep'], history['rewards/action'], color='green', alpha=.75, lw=2, label='Action Penalty')
plot_without_zeros(history['timestep'], history['rewards/distance'], color='blue', alpha=.75, lw=2, label='Distance Reward')
plot_without_zeros(history['timestep'], history['rewards/origin'], color='darkorange', alpha=.75, lw=2, label='Origin Reward')

# Stage ticks
unique, stage_idx = np.unique(history['stage'][::-1], return_index=True)
stage_idx = len(history['stage']) - stage_idx
stage_idx = stage_idx[:-1]
[ax.axvline(x=history['timestep'][idx], color='black', alpha=.5, linestyle='dashed', lw=1) for idx in stage_idx]

# Labels
ax.set_xlabel('Timestep')
ax.set_ylabel('Reward')
ax.legend(loc='lower right', ncols=3)

# Styling
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlim([0, history['timestep'].max()])

# Save plot
fname = f'{config["data"]["dataset"]}_performance.pdf'
fig.savefig(os.path.join(PLOT_FOLDER, fname), dpi=300)

# %% [markdown]
# ##### Comparison

# %%
# Method comparison
if 'integration' in memories:
    # Comparison metrics
    metric_rand = lambda X: np.random.rand()
    metric_silhouette = lambda X: sklearn.metrics.silhouette_score(X, labels)
    metric_ch_score = lambda X: sklearn.metrics.calinski_harabasz_score(X, labels)
    def metric_knn_ami(X):
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
        knn.fit(X, labels)
        pred = knn.predict(X)
        return sklearn.metrics.adjusted_mutual_info_score(labels, pred)

    metric_tuples = {
        'rand': (metric_rand, {'label': 'Random'}),
        'sc': (metric_silhouette, {'label': 'Silhouette Coefficient'}),
        'knn_ami': (metric_knn_ami, {'label': 'KNN Adjusted Mutual Information'}),
        'ch': (metric_ch_score, {'label': 'Calinski Harabasz Index', 'scale': 'log'}),
    }

    # Select metrics
    metric_x, kwargs_x = metric_tuples['ch']
    metric_y, kwargs_y = metric_tuples['knn_ami']

    # Get other methods
    method_dir = os.path.join(BASE_FOLDER, '../other_methods/runs', config['data']['dataset'])
    method_names = next(os.walk(method_dir))[1]
    method_results = {}
    for name in method_names:
        # Get output files
        files = os.listdir(os.path.join(method_dir, name))
        r = re.compile('^P\d+.txt$')
        files = list(filter(r.match, files))

        # Record
        for i, file in enumerate(files):
            proj = np.loadtxt(os.path.join(method_dir, name, file))
            method_results[(name, i)] = proj

    # Add cellTRIP
    method_results[('cellTRIP', -1)] = memories['integration']['states'][-1].detach().cpu()

    # Compile and calculate performances
    performance = pd.DataFrame(columns=['Method', 'Modality', 'x', 'y'])
    for key, data in method_results.items():
        performance.loc[performance.shape[0]] = [key[0], key[1], metric_x(data), metric_y(data)]

    # Plot with text
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, layout='constrained')
    method_colors = {}
    annotations = []
    for i, r in performance.iterrows():
        # Set color
        if r['Method'] not in method_colors: method_colors[r['Method']] = sns.color_palette()[len(method_colors)]
        
        # Plot
        ax.scatter(
            r['x'],
            r['y'],
            color=method_colors[r['Method']],
            s=100,
        )

        # Cross lines
        ax.axvline(x=r['x'], ls='--', alpha=.1, color='black', zorder=.3)
        ax.axhline(y=r['y'], ls='--', alpha=.1, color='black', zorder=.3)

        # Annotate
        text = f'{r["Method"]}' + (f' ({r["Modality"]})' if r['Modality'] != -1 else '')
        annotations.append(ax.text(
            r['x'], r['y'], text,
            ha='center', va='center', fontsize='large'))

    # Styling
    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax.set(
        **{'x'+k: v for k, v in kwargs_x.items()},
        **{'y'+k: v for k, v in kwargs_y.items()},
    )
    ax.axvline(x=0, ls='-', alpha=.6, color='black', zorder=.1)
    ax.axhline(y=0, ls='-', alpha=.6, color='black', zorder=.1)

    # Adjust Annotation Positions
    from adjustText import adjust_text
    adjust_text(
        annotations,
        expand=(1.2, 2),
        arrowprops=dict(arrowstyle='->', color='black', zorder=.3),
    )

    # Save plot
    fname = f'{config["data"]["dataset"]}_comparison.pdf'
    fig.savefig(os.path.join(PLOT_FOLDER, fname), dpi=300)

# %% [markdown]
# ### Dynamic Visualizations

# %%
# Prepare data
skip = 5
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
num_lines = 100
if analysis_key == 'temporal': num_lines *= len(temporal['stages'])
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
    'num_lines': num_lines,
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
fname =                                     f'{config["data"]["dataset"]}'
if stage_override is not None: fname +=     f'_{stage_override:02}'
fname +=                                    f'_{analysis_key}'
fname +=                                    f'.{file_type}'
ani.save(os.path.join(PLOT_FOLDER, fname), writer=writer, dpi=300)

# CLI
print()


