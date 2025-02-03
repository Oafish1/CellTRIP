# %%
from collections import defaultdict
import gzip
import os
import pickle
import re
import tempfile
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.neighbors
import torch
import wandb

# Enable text output in notebooks
import tqdm.auto
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.auto.tqdm
from tqdm import tqdm

import data
import celltrip

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
# - TODO
#   - Maybe filter loss plot to selected stage?
#   - Add outdir, datadir, etc.
#   - Get file location rather than `abspath`
#   - Perturbation analysis for all features, filtered to only important ones, as well as .txt output
#   - Synchronize colors for each method
#   - Add perturbation and trajectory
#   - Try using known reference points (i.e. positional data) to impute absolute, rather than relative, values
#   - Maybe add buffer to perturbation analysis start
#   - Add arguments like wandb username/project, etc. as well as local db

# %% [markdown]
# ## Arguments

# %%
# Arguments
import argparse
parser = argparse.ArgumentParser(description='Create a video of the specified CellTRIP model')

# Main parameters
group = parser.add_argument_group('General')
group.add_argument('run_id', type=str, help='Run ID from WandB to use for processing')
group.add_argument('analysis_key', choices=('convergence', 'discovery', 'temporal', 'perturbation'), nargs='+', type=str, help='Type of analyses to perform (one or more)')
group.add_argument('-S', '--seed', type=int, help='Override simulation seed')
group.add_argument('--gpu', default='0', type=str, help='GPU(s) to use')

# Model parameters
group = parser.add_argument_group('Simulation')
group.add_argument('-b', '--batch', metavar='MAX_BATCH', dest='max_batch', type=int, help='Override number of nodes which can calculate actions simultaneously')
group.add_argument('--num', metavar='NUM_NODES', dest='num_nodes', type=int, help='Override number of nodes to take from data')
group.add_argument('--nodes', metavar='NUM_NEIGHBORS', dest='max_nodes', type=int, help='Override neighbors considered by each node')
group.add_argument('--stage', type=int, help='Override model stage to use. 0 is random initialization')

# Simulation specific arguments
group = parser.add_argument_group('Analysis')
group.add_argument('--discovery_key', type=int, default=0, help='Type of discovery analysis (0: Auto)')
group.add_argument('--temporal_key', type=int, default=0, help='Type of temporal analysis (0: Auto, 1: TemporalBrain)')
group.add_argument('--force', action='store_true', help='Rerun analysis even if already stored in memory')

# Video parameters
group = parser.add_argument_group('Video')
group.add_argument('--novid', action='store_true', help='Skip video generation')
group.add_argument('-g', '--gif', action='store_true', help='Output as a GIF rather than MP4')
group.add_argument('-s', '--skip', type=int, default=5, help='Number of steps to advance each frame')
group.add_argument('--reduction', choices=('umap', 'pca', 'none'), default='pca', type=str, dest='reduction_type', help='Reduction type to use for high-dimensional projections in 3D visualization')
group.add_argument('--force_reduction', action='store_true', help='Force reduction, even if unnecessary')
group.add_argument('--reduction_batch', type=int, default=100_000, help='Max number of states to reduce in one computation')

# Legacy compatibility
group = parser.add_argument_group('Legacy Compatiiblity')
group.add_argument('--total_statistics', action='store_true', help='Compatibility argument to compute mean and variance over all samples')

# List of common runs
# 'brf6n6sn': TemporalBrain Random 100 Max
# 'rypltvk5': MMD-MA Random 100 Max (requires `total_statistics`)
# '32jqyk54': MERFISH Random 100 Max
# 'c8zsunc9': ISS Random 100 Max
# 'maofk1f2': ExSeq NR
# 'f6ajo2am': smFish NR
# 'vb1x7bae': MERFISH NR
# '473vyon2': ISS NR

# Notebook defaults and script handling
if not celltrip.utilities.is_notebook():
    args = parser.parse_args()
else:
    args = parser.parse_args('--novid --total_statistics rypltvk5 convergence discovery temporal perturbation'.split(' '))  # MMD-MA
    # args = parser.parse_args('--novid 32jqyk54 convergence discovery temporal perturbation'.split(' '))  # MERFISH
    # args = parser.parse_args('--novid c8zsunc9 convergence discovery temporal perturbation'.split(' '))  # ISS

# Set env vars
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

# %% [markdown]
# ## Load Data, Model, and Environment

# %%
# Load run
print(f'Loading run {args.run_id}')
api = wandb.Api()
run = api.run(f'oafish/CellTRIP/{args.run_id}')
config = defaultdict(lambda: {})
for k, v in run.config.items():
    dict_name, key = k.split('/')
    config[dict_name][key] = v
config = dict(config)

# Reproducibility
notebook_seed = args.seed if args.seed is not None else config['note']['seed']  # Potentially destructive if randomly sampled
# torch.use_deterministic_algorithms(True)
torch.manual_seed(notebook_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(notebook_seed)
np.random.seed(notebook_seed)

# Get latest policy
print('\tFinding model')
latest_mdl = [0, None]  # Pkl
latest_wgt = [0, None]  # State dict
# Compatibility with models of the previous naming convention
# for file in run.files():
#     matches = re.findall(f'^(?:models|trained_models)/policy_(\w+).(mdl|wgt)$', file.name)
#     if len(matches) > 0: stage = int(matches[0][0]); ftype = matches[0][1]
#     else: continue
#     if stage == 0: add_one = True; break
# else: add_one = False
# Iterate through model files
for file in run.files():
    matches = re.findall(f'^(?:models|trained_models)/policy_(\w+).(mdl|wgt)$', file.name)
    if len(matches) > 0: stage = int(matches[0][0]); ftype = matches[0][1]
    else: continue
    # if add_one: stage += 1

    # Record
    latest_known_stage = latest_mdl[0] if ftype == 'mdl' else latest_wgt[0]
    if (args.stage is None and stage > latest_known_stage) or (args.stage is not None and stage == args.stage):
        if ftype == 'mdl': latest_mdl = [stage, file]
        elif ftype == 'wgt': latest_wgt = [stage, file]
print(f'\t\tMDL policy found at stage {latest_mdl[0]}')
print(f'\t\tWGT policy found at stage {latest_wgt[0]}')

# Load data
print(f'\tLoading dataset {config["data"]["dataset"]}')
modalities, types, features = data.load_data(config['data']['dataset'], DATA_FOLDER)
# config['data'] = celltrip.utilities.overwrite_dict(config['data'], {'standardize': True})  # Old model compatibility
# config['data'] = celltrip.utilities.overwrite_dict(config['data'], {'top_variant': config['data']['pca_dim'], 'pca_dim': None})  # Swap PCA with top variant (testing)
if args.num_nodes is not None: config['data'] = celltrip.utilities.overwrite_dict(config['data'], {'num_nodes': args.num_nodes})
if args.max_batch is not None: config['train'] = celltrip.utilities.overwrite_dict(config['train'], {'max_batch': args.max_batch})
for k in ('standardize', 'pca_dim', 'top_variant'):
    # Legacy compatibility for missing default arguments
    if k not in config['data']: config['data'][k] = None
ppc = celltrip.utilities.Preprocessing(**config['data'], device=DEVICE)
modalities, features = ppc.fit_transform(modalities, features, total_statistics=args.total_statistics)
modalities, types = ppc.subsample(modalities, types)
modalities = ppc.cast(modalities)
labels = types[0][:, 0]
times = types[0][:, -1]

# Load env
env = celltrip.environments.trajectory(*modalities, **config['env'], **config['stages']['env'][0], device=DEVICE)
for weight_stage in config['stages']['env'][1:latest_mdl[0]+1]:
    env.set_rewards(weight_stage)
application_type = 'integration' if len(env.reward_distance_target) == len(modalities) else 'imputation'

# Load model file
load_type = 'WGT'
if load_type == 'MDL' and latest_mdl[0] != 0:
    print('\tLoading MDL model')
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_mdl[1].download(tmpdir, replace=True)
        policy = torch.load(os.path.join(tmpdir, latest_mdl[1].name))
elif load_type == 'WGT' and latest_wgt[0] != 0:
    print('\tLoading WGT model')
    # Mainly used in the case of old argument names, also generally more secure
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_wgt[1].download(tmpdir, replace=True)
        # config['policy'] = celltrip.utilities.overwrite_dict(config['policy'], {'positional_dim': 6, 'modal_dims': [76]})  # Old model compatibility
        if args.max_nodes is not None: config['policy'] = celltrip.utilities.overwrite_dict(config['policy'], {'max_nodes': args.max_nodes})
        policy = celltrip.models.PPO(**config['policy'])
        incompatible_keys = policy.load_state_dict(torch.load(os.path.join(tmpdir, latest_wgt[1].name), weights_only=True))
else:
    print('\tGenerating random model')
    # Use random model
    policy = celltrip.models.PPO(**config['policy'])
policy = policy.to(DEVICE).eval()
policy.actor.set_action_std(1e-7)

# %% [markdown]
# ## Run Simulation

# %% [markdown]
# #### Parameter Presets

# %%
# Choose key
optimize_memory = True  # Saves memory by shrinking env based on present, also fixes reward calculation for non-full present mask
perturbation_features = [np.random.choice(len(fs), 10, replace=False) for i, fs in enumerate(features) if (i not in env.reward_distance_target) or (len(env.reward_distance_target) == len(modalities))]

# Define matching state manager classes
state_manager_class = {
    'convergence': celltrip.utilities.ConvergenceStateManager,
    'discovery': celltrip.utilities.DiscoveryStateManager,
    'temporal': celltrip.utilities.TemporalStateManager,
    'perturbation': celltrip.utilities.PerturbationStateManager,
}

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
discovery = discovery[args.discovery_key]

# Stage order list
temporal = []
# Reverse alphabetical (ExSeq, MERFISH, smFISH, ISS, MouseVisual)
temporal_general = {'stages': [[l] for l in np.unique(times)[::-1]]}
temporal_temporalBrain = {'stages': [
    ['EaFet1'],
    ['EaFet2'],
    ['LaFet1'],
    ['LaFet2'],
    ['Inf1'],
    ['Inf2'],
    ['Child1'],
    ['Child2'],
    ['Adol1'],
    ['Adol2'],
    ['Adult1'],
    ['Adult2'],
]}
temporal += [temporal_general]
temporal += [temporal_temporalBrain]
# Choose stage order
temporal = temporal[args.temporal_key]

# Perturbation feature names
perturbation_feature_names = [[fnames[pf] for pf in pfs] for pfs, fnames in zip(perturbation_features, features)]

# Initialize memories
memories = {}

# %% [markdown]
# #### Generate Simulation

# %%
# Load memories
fname =                                     f'{args.run_id}'
if args.stage is not None: fname +=         f'_{args.stage:02}'
fname +=                                    f'_{config["data"]["dataset"]}'
fname +=                                    f'_memories.pkl.gzip'

# Load memories
if os.path.exists(fname):
    print('Loading existing memories')
    with gzip.open(fname, 'rb') as f: memories = pickle.load(f)

# Run simulation if needed
for ak in args.analysis_key:
    if ak not in memories or args.force:
        print(f'Running {ak} simulation')

        # Profiling
        profile = False
        if profile: torch.cuda.memory._record_memory_history(max_entries=100000)

        # Choose state manager
        state_manager = state_manager_class[ak](
            device=DEVICE,
            discovery=discovery,
            temporal=temporal,
            perturbation_features=perturbation_features,
            modal_targets=env.reward_distance_target,
            num_nodes=modalities[0].shape[0],
            dim=env.dim,
            # vel_threshold=1e-1,  # Temporal testing
        )

        # Utility parameters
        get_current_stage = lambda: (
            state_manager.current_stage
            if np.array([ak in akt for akt in ('temporal', 'perturbation')]).any()
            else -1
        )
        get_max_stage = lambda: (
            len(temporal['stages'])-1 if ak == 'temporal'
            else sum([len(pf) for pf in perturbation_features])+1 if ak == 'perturbation'
            else -1
        )
        # TODO: Make perturbation more memory-efficient
        use_modalities = np.array([ak in akt for akt in ('perturbation',)]).any()

        # Initialize
        env.set_modalities(modalities); env.reset(); memories[ak] = defaultdict(lambda: [])

        # Modify
        state_vars, end = state_manager(
            # present=present,
            state=env.get_state(),
            modalities=ppc.cast(ppc.inverse_transform(ppc.inverse_cast(modalities)), device='cpu') if use_modalities else modalities,
            labels=labels,
            times=times,
        )
        present = state_vars['present']
        memory_mask = present if optimize_memory else torch.ones_like(present, device=DEVICE)
        full_state = state_vars['state']
        env.set_state(full_state[memory_mask])
        raw_modalities = state_vars['modalities']
        processed_modalities = [m[memory_mask.cpu()] for m in raw_modalities]
        if use_modalities: processed_modalities = ppc.cast(ppc.transform(ppc.inverse_cast(processed_modalities)))
        env.set_modalities(processed_modalities)

        # Continue initializing
        memories[ak]['present'].append(present.cpu())
        memories[ak]['states'].append(full_state.cpu())
        memories[ak]['stages'].append(get_current_stage())
        memories[ak]['rewards'].append(torch.zeros(modalities[0].shape[0]))

        # Simulate
        get_desc = lambda ts, st: f'\tTimestep {ts}' + (f', Stage {st+1}/{get_max_stage()+1}' if st != -1 else '')
        timestep = 0; pbar = tqdm(ascii=True, desc=get_desc(timestep, get_current_stage()), ncols=100)  # CLI
        while True:
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
                processed_modalities = [m[memory_mask.cpu()] for m in raw_modalities]
                if use_modalities: processed_modalities = ppc.cast(ppc.transform(ppc.inverse_cast(processed_modalities)))
                env.set_modalities(processed_modalities)

            # Record
            memories[ak]['present'].append(present.cpu())
            memories[ak]['states'].append(full_state.cpu())
            memories[ak]['stages'].append(get_current_stage())
            memories[ak]['rewards'].append(rewards.cpu())

            # CLI
            timestep += 1
            update_timestep = 10
            if timestep % update_timestep == 0:
                pbar.set_description(get_desc(timestep, get_current_stage()))
                pbar.update(update_timestep)

            # End
            if end: break

        # CLI
        pbar.close()

        # Stack
        memories[ak]['present'] = torch.stack(memories[ak]['present'])
        memories[ak]['states'] = torch.stack(memories[ak]['states'])
        memories[ak]['stages'] = torch.tensor(memories[ak]['stages'])
        memories[ak]['rewards'] = torch.stack(memories[ak]['rewards'])
        memories[ak] = dict(memories[ak])

        # Profiling
        if profile:
            torch.cuda.memory._dump_snapshot('cuda_profile.pkl')
            torch.cuda.memory._record_memory_history(enabled=None)

        # Save into half-accuracy gzip (8,506 KB -> 3,236 KB)
        print(f'\tSaving memories')
        compressed_type = torch.float16
        with gzip.open(fname, 'wb') as f:
            func_attr = lambda attr: attr.type(compressed_type) if attr.dtype not in (torch.long, torch.bool) else attr
            celltrip.utilities.dict_map(memories[ak], func_attr, inplace=True)
            pickle.dump(memories, f)

# %% [markdown]
# #### Memory Summary

# %%
# Statistics
for ak in args.analysis_key:
    print(f'Statistics {ak}')

    ## Stages
    stages, counts = np.unique(memories[ak]['stages'], return_counts=True)
    print('\tSteps per Stage')
    for s, c in zip(stages, counts):
        print(f'\t\t\t{s}\t{c}')
        
    ## Memory
    print('\tCompressed Memory Sizes')
    for k in memories[ak]:
        t_size = sum([t.element_size() * t.nelement() if isinstance(t, torch.Tensor) else 64/8 for t in memories[ak][k]]) / 1024**3
        print(f'\t\t\t{k} size\t{t_size:.3f} Gb')

    ## Performance
    print(f'\tAverage Reward: {memories[ak]["rewards"].cpu().mean():.3f}')

# %% [markdown]
# ## Static Analyses

# %%
print('Plotting static analyses')

# %% [markdown]
# #### Loss Plot

# %%
print('\tTraining rewards')

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
fname = f'{args.run_id}_{config["data"]["dataset"]}_loss.pdf'
fig.savefig(os.path.join(PLOT_FOLDER, fname), dpi=300)

# %% [markdown]
# #### Integration Performance Comparison

# %%
def get_other_methods(prefix):
    # Get other methods
    method_results = {}
    try:
        method_dir = os.path.join(BASE_FOLDER, '../other_methods/runs', config['data']['dataset'])
        method_names = next(os.walk(method_dir))[1]
    except: method_names = []
    for name in method_names:
        # Get output files
        files = os.listdir(os.path.join(method_dir, name))
        r = re.compile(f'^{prefix}(\d+)(?:_(\d+))?.txt$')
        files = list(filter(r.match, files))

        # Record
        for file in files:
            modality, seed = r.match(file)[1], r.match(file)[2]
            method_results[(name, modality, seed)] = os.path.join(method_dir, name, file)

    return method_results

# %%
# Comparison metrics
metric_rand = lambda X: np.random.rand()
metric_silhouette = lambda X: sklearn.metrics.silhouette_score(X, labels)
metric_ch_score = lambda X: sklearn.metrics.calinski_harabasz_score(X, labels)
def metric_knn_ami(X):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, labels)
    pred = knn.predict(X)
    return sklearn.metrics.adjusted_mutual_info_score(labels, pred)

# Metric metadata
metric_tuples = {
    'rand': (metric_rand, {'label': 'Random'}),
    'sc': (metric_silhouette, {'label': 'Silhouette Coefficient'}),
    'knn_ami': (metric_knn_ami, {'label': 'KNN Adjusted Mutual Information'}),
    'ch': (metric_ch_score, {'label': 'Calinski Harabasz Index', 'scale': 'log'}),
}

# Metric selection per analysis type
comparison_dict = {
    'integration': {
        'prefix': 'P',
        'metrics': (metric_tuples['ch'], metric_tuples['knn_ami']),
    }
}

# Integration method comparison
desired_application = 'integration'
if 'convergence' in args.analysis_key and application_type == desired_application:
    print(f'\tComparison Integration')

    # Select metrics
    (metric_x, kwargs_x), (metric_y, kwargs_y) = comparison_dict[desired_application]['metrics']

    # Get other methods
    method_results = get_other_methods(comparison_dict[desired_application]["prefix"])

    # Add CellTRIP
    method_results[('CellTRIP', '-1', notebook_seed)] = memories['convergence']['states'][-1].detach().cpu()

    # Compile and calculate performances
    raw_performance = pd.DataFrame(columns=['Method', 'Modality', 'Seed', 'x', 'y'])
    for key, fname in method_results.items():
        method, modality, seed = key
        if method == 'CellTRIP': data = fname
        else: data = np.loadtxt(fname)
        raw_performance.loc[raw_performance.shape[0]] = [*key, metric_x(data), metric_y(data)]

    # Aggregate to group statistics
    group = raw_performance.groupby(['Method', 'Modality'])
    group_mean = group[['x', 'y']].mean().rename(columns=lambda n: f'{n}_mean')
    group_var = group[['x', 'y']].var().rename(columns=lambda n: f'{n}_var')
    group_count = group[['x']].count().rename(columns={'x': 'Count'})
    performance = group_mean.join(group_var).join(group_count).fillna(0).reset_index()

    # Print statistics
    print(f'\t\t{"Method":<10}\tModal\tx Mean\tx Var\ty Mean\ty Var\tCount')
    for i, r in performance.sort_values('Modality').iterrows():
        print(f'\t\t{r["Method"]:<8}\t{r["Modality"]}\t{r["x_mean"]:.3f}\t{r["x_var"]:.3f}\t{r["y_mean"]:.3f}\t{r["y_var"]:.3f}\t{r["Count"]}')

    # Plot with text
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, layout='constrained')
    method_colors = {}
    annotations = []
    for i, r in performance.iterrows():
        # Set color
        if r['Method'] not in method_colors: method_colors[r['Method']] = sns.color_palette()[len(method_colors)]
        
        # Plot
        ax.scatter(
            r['x_mean'],
            r['y_mean'],
            color=method_colors[r['Method']],
            s=100,
        )

        # Cross lines
        ax.plot([r['x_mean']-r['x_var'], r['x_mean']+r['x_var']], 2*[r['y_mean']], ls='--', color='gray', zorder=.3)
        ax.plot(2*[r['x_mean']], [r['y_mean']-r['y_var'], r['y_mean']+r['y_var']], ls='--', color='gray', zorder=.3)

        # Annotate
        text = f'{r["Method"]}' + (f' ({r["Modality"]})' if r['Modality'] != -1 else '')
        annotations.append(ax.text(
            r['x_mean'], r['y_mean'], text,
            ha='center', va='center', fontsize='large'))

    # Styling
    # ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # ax.axvline(x=0, ls='-', alpha=.6, color='black', zorder=.1)
    # ax.axhline(y=0, ls='-', alpha=.6, color='black', zorder=.1)
    ax.set(
        **{'x'+k: v for k, v in kwargs_x.items()},
        **{'y'+k: v for k, v in kwargs_y.items()},
    )
    ax.tick_params(axis='both', which='both', bottom=True, left=True)

    # Adjust Annotation Positions
    from adjustText import adjust_text
    adjust_text(
        annotations,
        # add_objects=ax.get_children()[0],
        expand=(2, 3), 
        arrowprops=dict(
            arrowstyle='-|>',
            mutation_scale=10,
            shrinkA=2, shrinkB=7,
            color='black',
        ),
    )

    # Save plot
    fname =                                     f'{args.run_id}'
    if args.stage is not None: fname +=         f'_{args.stage:02}'
    fname +=                                    f'_{config["data"]["dataset"]}'
    fname +=                                    f'_comparison'
    fname +=                                    f'_{desired_application}.pdf'
    fig.savefig(os.path.join(PLOT_FOLDER, fname), dpi=300)

# %% [markdown]
# #### Imputation Performance Comparison

# %%
# Single modality imputation
if 'convergence' in args.analysis_key and application_type == 'imputation' and len(env.reward_distance_target) == 1:
    # Get other methods
    method_results = get_other_methods('I')

    # Add CellTRIP
    raw_celltrip = memories['convergence']['states'][-1].detach().cpu()[:, :env.dim]
    # shrunk_celltrip = sklearn.decomposition.PCA(n_components=modalities[env.reward_distance_target[0]].shape[1]).fit_transform(raw_celltrip)
    method_results[('CellTRIP', env.reward_distance_target[0]+1, notebook_seed)] = raw_celltrip

    # Calculate modal dist
    # raw_modalities = ppc.cast(ppc.inverse_transform(ppc.inverse_cast(modalities)))
    modal_dist = [celltrip.utilities.euclidean_distance(m) for m in modalities]

    # Compile and calculate performances
    raw_performance = None
    for key, fname in method_results.items():
        method, modality, seed = key
        if method == 'CellTRIP': data = fname
        else: data = np.loadtxt(fname)
        data = torch.Tensor(data)

        # Scale results
        if method == 'CellTRIP': data = data * np.sqrt(data.shape[1])  # TODO: See if this is justified
        # if not method == 'CellTRIP': data = data / ppc.standardize_std[int(modality)-1]

        # Compute error
        data_dist = celltrip.utilities.euclidean_distance(data)
        sample_mse = (data_dist - modal_dist[int(modality)-1].cpu()).square().mean(dim=-1)
        feature_mse = (data_dist - modal_dist[int(modality)-1].cpu()).square().mean(dim=0)

        # Record
        df = pd.DataFrame({'Method': method, 'Modality': modality, 'Seed': seed, 'Metric': 'Sample Inter-Cell MSE', 'Value': sample_mse})
        if raw_performance is None: raw_performance = df
        else: raw_performance = pd.concat((raw_performance, df), ignore_index=True, axis=0)

    # Fill NA (For non-random methods)
    raw_performance = raw_performance.fillna(0)

    # Plot performances
    main_metric = 'Sample Inter-Cell MSE'

    # Filter to only the best result from each method
    best_seeds = (
        raw_performance.loc[raw_performance['Metric'] == main_metric]
        .groupby(['Method', 'Modality', 'Seed'])[['Value']].mean().reset_index()
        .sort_values('Value', ascending=False).groupby(['Method', 'Modality'])[['Seed']].first().reset_index().to_numpy()
    )
    mask = np.zeros(raw_performance.shape[0], dtype=bool)
    for idx in best_seeds:
        mask += (raw_performance.to_numpy()[:, :3] == idx).all(axis=-1)
    filtered_performance = raw_performance.iloc[mask]

    # Generate visuals
    filtered_performance = filtered_performance.loc[filtered_performance['Metric'] == main_metric]
    order = filtered_performance['Method'].unique()
    fig, ax = plt.subplots(1, 1, figsize=(3, 6), sharex=True, layout='constrained')
    sns.boxplot(
        data=filtered_performance,
        x='Method', y='Value', hue='Method',
        order=order,
        ax=ax)
    with warnings.catch_warnings(record=False) as w:
        warnings.simplefilter('ignore')
        sns.stripplot(
            data=filtered_performance.sample(frac=.05),
            x='Method', y='Value', hue='Method',
            order=order,
            size=2, palette=sns.color_palette(['black']), legend=False, dodge=False, ax=ax)

    # Styling
    ax.set(title=main_metric, xlabel=None, ylabel=None)
    ax.tick_params(axis='both', which='both', bottom=False, left=True)
    ax.set_yscale('log')

    # Xlabels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center', va='baseline')
    max_height = max([l.get_window_extent(renderer=ax.figure.canvas.get_renderer()).height for l in ax.get_xticklabels()])
    fontsize = ax.get_xticklabels()[0].get_size()
    pad = fontsize / 2 + max_height / 2
    ax.tick_params(axis='x', pad=pad)

    # Save plot
    fname =                                     f'{args.run_id}'
    if args.stage is not None: fname +=         f'_{args.stage:02}'
    fname +=                                    f'_{config["data"]["dataset"]}'
    fname +=                                    f'_comparison'
    fname +=                                    f'_imputation.pdf'
    fig.savefig(os.path.join(PLOT_FOLDER, fname), dpi=300)

# %% [markdown]
# #### Feature Importance

# %%
# Perturbation significance analysis
if 'perturbation' in memories:
    print('\tFeature effect size')
    
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

        # Record steady state after convergence
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
        print(f'\t\tModality {j}')
        for pf, pfn in zip(pfs, pfns):
            print(f'\t\t\t{pfn:<15}{effect_sizes[i]:.02e}')
            i += 1

# %% [markdown]
# ## Dynamic Visualizations

# %%
print('Plotting dynamic visualizations')

# %% [markdown]
# #### Video

# %%
for ak in args.analysis_key:
    if args.novid: break
    print(f'\tVideo {ak}')

    # Prepare data
    present = memories[ak]['present'].cpu()
    states = memories[ak]['states'].cpu()
    stages = memories[ak]['stages'].cpu()
    rewards = memories[ak]['rewards'].cpu()
    base_env = celltrip.environments.trajectory(*[torch.empty((0, 0)) for _ in range(len(modalities))], **config['env'])

    # Testing for portions of large datasets
    # sub_idx = np.random.choice(modalities[0].shape[0], 1_000, replace=False)
    # modalities, labels, times = [m[sub_idx] for m in modalities], labels[sub_idx], times[sub_idx]
    # present, states, rewards = present[:, sub_idx], states[:, sub_idx], rewards[:, sub_idx]

    # Testing for larger dims
    # states = torch.concatenate((states, states), dim=-1)
    # base_env.dim *= 2

    # Skip data
    present, states, stages, rewards = present[::args.skip], states[::args.skip], stages[::args.skip], rewards[::args.skip]
    # if states_3d is not None: states_3d = states_3d[::args.skip]

    # Reduce dimensions
    if states.shape[-1] > 2*3 or args.force_reduction:
        print('\t\tReducing state dimensionality')
        # Get idx of last state in designated stage
        stage_unique, stage_idx = np.unique(stages.numpy()[::-1], return_index=True)
        stage_idx = memories[ak]['stages'].shape[0] - stage_idx - 1

        # Choose reduction type
        if args.reduction_type == 'umap':
            import umap
            fit_reducer = lambda data: umap.UMAP(n_components=3, random_state=notebook_seed).fit(data)
            transform_reducer = lambda reducer, data: torch.Tensor(reducer.transform(data))
        elif args.reduction_type == 'pca':
            import sklearn.decomposition
            fit_reducer = lambda data: sklearn.decomposition.PCA(n_components=3, random_state=notebook_seed).fit(data)
            transform_reducer = lambda reducer, data: torch.Tensor(reducer.transform(data))
        elif args.reduction_type is None or args.reduction_type == 'none':
            initialize_reducer = lambda: None
            transform_reducer = lambda reducer, data: data

        # Get steady state
        if ak in ('convergence', 'discovery', 'perturbation',):
            reducer = fit_reducer(states[stage_idx[0]])
            get_reducer = lambda stage: reducer
        elif ak in ('temporal',):
            get_reducer = lambda stage: fit_reducer(states[stage_idx[stage]])

        # UMAP
        get_desc = lambda stage: f'\t\t\tProjecting ({stage}/{stage_unique.max()})'
        states_3d = []; pbar = tqdm(total=states.shape[0]*states.shape[1], desc=get_desc(0), ascii=True, ncols=100)
        for stage in stage_unique:
            pbar.set_description(get_desc(stage))
            stage_states = states[stages==stage].reshape((-1, states.shape[-1]))
            for i in range(0, stage_states.shape[0], args.reduction_batch):
                states_3d.append(transform_reducer(get_reducer(stage), stage_states[i:i+args.reduction_batch]))
                pbar.update(stage_states[i:i+args.reduction_batch].shape[0])
        pbar.close()
        states_3d = torch.concatenate(states_3d, dim=0).reshape((*states.shape[:-1], 3))
        states_3d = torch.concatenate((states_3d, torch.zeros_like(states_3d)), dim=-1)
    else:
        states_3d = None

    # CLI
    print('\t\tGenerating video')

    # Parameters
    interval = 1e3*env.delta/3  # Time between frames (3x speedup)
    min_max_vel = 1e-2 if ak in ('convergence', 'discovery') else -1  # Stop at first frame all vels are below target. 0 for full play
    frame_override = None  # Manually enter number of frames to draw
    rotations_per_second = .1  # Camera azimuthal rotations per second
    num_lines = 100
    if ak == 'temporal': num_lines *= len(temporal['stages'])**2

    # Create plot based on key
    # NOTE: Standard 1-padding all around and between figures
    # NOTE: Left, bottom, width, height
    if ak in ('convergence', 'discovery'):
        figsize = (15, 10)
        fig = plt.figure(figsize=figsize)
        axs = [
            fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
            fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
            fig.add_axes([10 /figsize[0], 1 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
        ]
        views = [
            celltrip.utilities.View3D,
            celltrip.utilities.ViewTemporalScatter,
            celltrip.utilities.ViewSilhouette,
        ]

    elif ak == 'temporal':
        figsize = (15, 10)
        fig = plt.figure(figsize=figsize)
        axs = [
            fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
            fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
            fig.add_axes([10 /figsize[0], 1 /figsize[1], 4 /figsize[0], 3.5 /figsize[1]]),
        ]
        views = [
            celltrip.utilities.View3D,
            celltrip.utilities.ViewTemporalScatter,
            celltrip.utilities.ViewTemporalDiscrepancy,
        ]

    elif ak in ('perturbation',):
        figsize = (20, 10)
        fig = plt.figure(figsize=figsize)
        axs = [
            fig.add_axes([1 /figsize[0], 1 /figsize[1], 8 /figsize[0], 8 /figsize[1]], projection='3d'),
            fig.add_axes([10 /figsize[0], 5.5 /figsize[1], 8 /figsize[0], 3.5 /figsize[1]]),
            fig.add_axes([10 /figsize[0], 1 /figsize[1], 3.5 /figsize[0], 3.5 /figsize[1]]),
            fig.add_axes([14.5 /figsize[0], 1 /figsize[1], 3.5 /figsize[0], 3.5 /figsize[1]]),
        ]
        views = [
            celltrip.utilities.View3D,
            celltrip.utilities.ViewPerturbationEffect,
            celltrip.utilities.ViewTemporalScatter,
            celltrip.utilities.ViewSilhouette,
        ]

    # Initialize views
    arguments = {
        # Data
        'present': present,
        'states': states,
        'states_3d': states_3d,
        'stages': stages,
        'rewards': rewards,
        'modalities': modalities,
        'labels': labels,
        # Data params
        'dim': base_env.dim,
        'modal_targets': base_env.reward_distance_target,
        'temporal_stages': temporal['stages'],
        'perturbation_features': perturbation_features,
        'perturbation_feature_names': perturbation_feature_names,
        'partitions': times if ak in ('temporal',) else None,
        # Arguments
        'interval': interval,
        'skip': args.skip,
        'seed': notebook_seed,
        # Styling
        'num_lines': num_lines,
        'ms': 5,  # 3
        'lw': 1,
    }
    views = [view(**arguments, ax=ax) for view, ax in zip(views, axs)]

    # Compile animation
    frames = states[..., env.dim:env.dim+3].square().sum(dim=-1).sqrt().max(dim=-1).values < min_max_vel
    frames = np.array([(frames[i] or frames[i+1]) if i != len(frames)-1 else frames[i] for i in range(len(frames))])  # Disregard interrupted sections of low movement
    frames = np.argwhere(frames)
    frames = frames[0, 0].item()+1 if len(frames) > 0 else states.shape[0]
    frames = frames if frame_override is None else frame_override

    # Update function
    pbar = tqdm(ascii=True, total=frames+1, desc='\t\t\tRendering', ncols=100)  # CLI, runs frame 0 twice
    def update(frame):
        # Update views
        for view in views:
            view.update(frame)

        # CLI
        update_timestep = 1
        if frame % update_timestep == 0:
            pbar.update(update_timestep)

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
    # NOTE: Requires `sudo apt-get install ffmpeg`
    file_type = 'mp4' if not args.gif else 'gif'
    if file_type == 'mp4': writer = animation.FFMpegWriter(fps=int(1e3/interval), extra_args=['-vcodec', 'libx264'], bitrate=8e3)  # Faster
    elif file_type == 'gif': writer = animation.FFMpegWriter(fps=int(1e3/interval))  # Slower
    fname =                                     f'{args.run_id}'
    if args.stage is not None: fname +=         f'_{args.stage:02}'
    fname +=                                    f'_{config["data"]["dataset"]}'
    fname +=                                    f'_{ak}'
    fname +=                                    f'.{file_type}'
    ani.save(os.path.join(PLOT_FOLDER, fname), writer=writer, dpi=300)

    # CLI
    pbar.close()

# %%
print('Done\n')


