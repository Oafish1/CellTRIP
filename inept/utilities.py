from collections import deque
from itertools import product
from time import perf_counter
import tracemalloc
import warnings

import mpl_toolkits.mplot3d as mp3d
import numpy as np
import scipy.sparse
import sklearn.decomposition
import torch


def cosine_similarity(a):
    # Calculate cosine similarity
    a_norm = a / a.norm(dim=1, keepdim=True)
    a_cos = a_norm @ a_norm.T

    return a_cos


def euclidean_distance(a, scaled=False):
    # Calculate euclidean distance
    dist = torch.cdist(a, a, p=2)
    # Scaled makes this equivalent to MSE
    if scaled: dist /= np.sqrt(a.shape[1])
    return dist


class time_logger():
    """Class made for easy logging with toggleable verbosity"""
    def __init__(
        self,
        discard_first_sample=False,
        record=True,
        verbose=False,
        memory_usage=False,
    ):
        self.discard_first_sample = discard_first_sample
        self.record = record
        self.verbose = verbose
        self.memory_usage = memory_usage

        self.history = {}
        self.start_time = perf_counter()

        if memory_usage:
            self.history_mem = {}
            tracemalloc.start()

    def log(self, str=''):
        """Print with message ``str`` if verbose.  Otherwise, skip"""
        if not (self.verbose or self.record):
            return  # Cut timing for optimization
        self.end_time = perf_counter()

        # Perform any auxiliary operations here
        time_elapsed = self.end_time - self.start_time
        # Record time
        if self.record:
            if str not in self.history:
                self.history[str] = []
            self.history[str].append(time_elapsed)
        if self.verbose:
            print(f'{str}: {time_elapsed}')
        # Record memory
        if self.memory_usage:
            if self.record:
                if str not in self.history_mem:
                    self.history_mem[str] = []
                self.history_mem[str].append(tracemalloc.get_traced_memory())
            if self.verbose:
                print(f'{str} Memory: Stored {self.history_mem[-1][0]} - Peak {self.history_mem[-1][1]}')
            tracemalloc.stop()

        # Re-time to avoid extra runtime cost
        self.start_time = perf_counter()
        if self.memory_usage:
            # WARNING, does not end tracemalloc
            tracemalloc.start()

    def aggregate(self, method='mean'):
        """Print mean times for all keys in ``self.history``"""
        running_total = 0
        for k, v in self.history.items():
            avg_time_elapsed = np.array(v)
            if self.discard_first_sample:
                avg_time_elapsed = avg_time_elapsed[1:]
            if method == 'mean':
                avg_time_elapsed = np.mean(np.array(v))
            elif method == 'sum':
                avg_time_elapsed = np.sum(np.array(v))

            running_total += avg_time_elapsed
            print(f'{k}: {avg_time_elapsed}')
            if self.memory_usage:
                stored = 0
                peak = 0
                for val in self.history_mem[k]:
                    stored += val[0]
                    if val[1] > peak:
                        peak = val[1]
                print(f'{k} Memory: Stored {stored} - Peak {peak}')
        print(f'Total: {running_total}')


class EarlyStopping:
    """
    Early stopping class with a few trigger methods

    Methods
    -------
    'absolute': Takes absolute best values for comparison and thresholding
    'average': Takes average values over a sliding window
    """
    def __init__(
        self,
        # Global parameters
        method='average',
        buffer=30,
        delta=1e-3,
        decreasing=False,

        # `average` method parameters
        window_size=15,

        # `absolute` method parameters
        # ...
    ):
        # Global parameters
        self.method = method
        self.buffer = buffer
        self.delta = delta
        self.decreasing = decreasing

        # `average` method parameters
        self.window_size = window_size
        self.history = deque([], window_size)

        # `absolute` method parameters
        # ...

        # Checks
        if self.method not in ('absolute', 'average'):
            raise ValueError(f'`{self.method}` not found in methods')

        # Initialize
        self.reset()

    def __call__(self, objective):
        """Return `True` if stop, else `False`."""
        # Record observation
        self.record_observation(objective)

        # Exit if not ready yet
        if self.current is None: return False

        # First observation
        if self.best is None: self.set_best(self.current)

        # Check for new best
        if self.decreasing and (self.current < self.threshold): self.set_best(self.current)
        elif not self.decreasing and (self.current > self.threshold): self.set_best(self.current)
        else: self.lapses += 1

        # Check lapses
        if self.lapses >= self.buffer:
            return True
        return False

    def record_observation(self, objective):
        "Record observation into internal `current` var based on method"
        if self.method == 'absolute':
            self.current = objective

        elif self.method == 'average':
            self.history.append(objective)
            if len(self.history) >= self.window_size:
                self.current = np.mean(self.history)

        else: raise ValueError(f'`{self.method}` not found in methods')

    def reset(self):
        "Reset class to default state"
        # State variables
        self.history.clear()
        self.current = None
        self.best = None
        self.threshold = None
        self.lapses = 0

    def set_best(self, objective):
        "Set current best"
        self.best = objective
        self.calculate_threshold()
        self.lapses = 0

    def calculate_threshold(self):
        "Calculate threshold for improvement"
        self.threshold = self.best + (-1 if self.decreasing else 1) * self.delta


def clean_return(ret, keep_array=False):
    "Clean return output for improved parsing"
    if not keep_array and len(ret) == 1: return ret[0]
    return ret


def split_state(
    state,
    idx=None,
    max_nodes=None,
    sample_strategy='random',
    return_mask=False,
):
    "Split full state matrix into individual inputs, idx is an optional array"
    # Parameters
    if idx is None: idx = list(range(state.shape[0]))
    if not isinstance(idx, list): idx = [idx]

    # Get self features for each node
    self_entity = state[idx]

    # Get node features for each state
    node_mask = torch.zeros((len(idx), state.shape[0]), dtype=torch.bool)
    for i, j in enumerate(idx): node_mask[i, j] = True
    node_mask = ~node_mask

    # Enforce max nodes
    num_nodes = state.shape[0] - 1
    use_mask = max_nodes is not None and max_nodes < num_nodes
    if use_mask:
        # Random sample `num_nodes` to `max_nodes`
        # NOTE: Not reproducible between forward and backward, at the moment
        if sample_strategy == 'random':
            # Filter nodes to `max_nodes` per idx
            num_nodes = max_nodes - 1
            probs = torch.rand_like(node_mask, dtype=torch.get_default_dtype())
            probs[~node_mask] = 0
            selected_idx = probs.argsort(dim=-1)[..., -num_nodes:]  # Take `num_nodes` highest values

            # Create new mask
            node_mask = torch.zeros((len(idx), state.shape[0]), dtype=torch.bool)
            for i in range(node_mask.shape[0]):
                node_mask[i, selected_idx[i]] = True

        # Sample closest nodes
        elif sample_strategy == 'closest':
            # TODO
            pass

    # Final formation
    node_entities = state.unsqueeze(0).expand(len(idx), *state.shape)
    node_entities = node_entities[node_mask].reshape(len(idx), num_nodes, state.shape[1])

    # Return
    ret = (self_entity, node_entities)
    if return_mask: ret += (node_mask,)
    return ret


def is_list_like(l):
    "Test if `l` has the `__len__` method`"
    try:
        len(l)
        return True
    except:
        return False


def recursive_tensor_func(input, func):
    "Applies function to tensors recursively"
    if type(input) == torch.Tensor:
        return func(input)
    
    return [recursive_tensor_func(member, func) for member in input]


def dict_map(dict, func, inplace=False):
    "Takes input dict `dict` and applies function to all members"
    if inplace:
        for k in dict: dict[k] = func(dict[k])
        return dict
    else:
        return {k: func(v) for k, v in dict.items()}
    

def dict_map_recursive_tensor_idx_to(dict, idx, device):
    "Take `idx` and cast to `device` from tensors inside of a dict, recursively"
    # NOTE: List indices into tensors create a copy
    # TODO: Add slice compatibility
    # Define function for each member
    def subfunc(x):
        if idx is not None: x = x[idx]
        if device is not None: x = x.to(device)
        return x
    
    # Apply and return (if needed)
    dict = dict_map(dict, lambda x: recursive_tensor_func(x, subfunc))
    return dict


class Sampler:
    "Sampler for data from `AdvancedMemoryBuffer`"
    def __init__(self, memory, rewards, sizes, mem_stage, gpu_stage, device):
        # Constants
        self.stage_dict = {
            'maxbatch': 0,
            'batch': 1,
            'minibatch': 2,
        }

        # Parameters and data
        self.memory = memory
        self.rewards = rewards
        self.sizes = sizes
        self.mem_stage = self.stage_dict[mem_stage]
        self.gpu_stage = self.stage_dict[gpu_stage]
        self.device = device

        # Variables
        self.idxs = [[] for _ in range(len(self.sizes))]
        self.mem_data = None
        self.mem_rewards = None
        self.gpu_data = None
        self.gpu_rewards = None

    def sample(self, stage, idx=None):
        # Determine where to sample from
        if stage == 0:
            # First stage samples from all memories
            sample_from = len(self.memory)
        elif (stage - 1) in (self.mem_stage, self.gpu_stage):
            # Stage after loading into memory resets indexes
            sample_from = len(self.idxs[stage-1])
        else:
            # Otherwise, sample from previous idx
            sample_from = self.idxs[stage - 1]

        # Set idx manually
        if idx is not None:
            # Make list if int
            if type(sample_from) == int: sample_from = np.array(range(sample_from))
            # Sample
            self.idxs[stage] = sample_from[idx]

        # Randomly sample idx
        self.idxs[stage] = np.random.choice(sample_from, self.sizes[stage])
        
    def act(self, stage):
        if stage == self.mem_stage:
            # Load into memory
            self.mem_data = self.memory[self.idxs[stage]]
            self.mem_rewards = self.rewards[self.idxs[stage]]
        
        if stage == self.gpu_stage:
            # Load into GPU
            if stage == self.mem_stage:
                self.gpu_data = dict_map_recursive_tensor_idx_to(self.mem_data, None, self.device)
                self.gpu_rewards = self.mem_rewards.to(self.device)
            else:
                self.gpu_data = dict_map_recursive_tensor_idx_to(self.mem_data, self.idxs[stage], self.device)
                self.gpu_rewards = self.mem_rewards[self.idxs[stage]].to(self.device)

        # Return latest data
        # print(stage)
        # torch.cuda.synchronize()
        # torch.cuda.reset_peak_memory_stats()
        # print(f'{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
        if stage >= self.gpu_stage:
            if stage == self.gpu_stage:
                data = self.gpu_data
                rewards = self.gpu_rewards
            else:
                data = dict_map_recursive_tensor_idx_to(self.gpu_data, self.idxs[stage], None)
                rewards = self.gpu_rewards[self.idxs[stage]]
        elif stage >= self.mem_stage:
            if stage == self.mem_stage:
                data = self.mem_data
                rewards = self.mem_rewards
            else:
                data = dict_map_recursive_tensor_idx_to(self.mem_data, self.idxs[stage], None)
                rewards = self.mem_rewards[self.idxs[stage]]
        else:
            data = None
            rewards = self.rewards[self.idxs[stage]]
        # torch.cuda.synchronize()
        # torch.cuda.reset_peak_memory_stats()
        # print(f'{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
        # print()

        return data, rewards

    def stage(self, stage, **kwargs):
        # Convert stage to int
        stage = self.stage_dict[stage]

        # Perform actions
        self.sample(stage, **kwargs)
        return self.act(stage)
    

class MemoryBuffer:
    """
    Naïve implementation of memory
    """
    def __init__(self):
        self.keys = []  # Entity key
        self.states = []
        self.actions = []
        self.action_logs = []
        self.state_vals = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.keys)

    def propagate_rewards(self, gamma=.99):
        # Propagate rewards backwards with decay
        rewards = []
        running_rewards = {k: 0 for k in np.unique(self.keys)}
        for key, reward, is_terminal in zip(self.keys[::-1], self.rewards[::-1], self.is_terminals[::-1]):
            if is_terminal: running_rewards[key] = 0  # Reset at terminal state
            running_rewards[key] = reward + gamma * running_rewards[key]
            rewards.append(running_rewards[key])
        rewards = rewards[::-1]
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return rewards

    def clear(self):
        del self.keys[:]
        del self.states[:]
        del self.actions[:]
        del self.action_logs[:]
        del self.state_vals[:]
        del self.rewards[:]
        del self.is_terminals[:]


def print_cuda_memory(peak=True):
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()
    print(f'{mem / 1024**3:.2f} GB')


class RunningStatistics:
    # https://github.com/fredlarochelle/RunningStats/blob/main/src/RunningStats.cpp
    def __init__(self, n_set=None, **kwargs):
        # Defaults
        self.reset(**kwargs)

        # Params
        self.n_set = n_set

    def reset(self, mean=0, m2=0):
        self.mean_x = mean
        self.m2 = m2
        self.n = 0

    def update(self, x):
        self.n += 1
        n = self.n_set if self.n_set is not None else self.n
            
        delta = x - self.mean_x
        self.mean_x += delta / n
        self.m2 += delta * (x - self.mean_x)

    def mean(self):
        return self.mean_x
    
    def variance(self):
        n = self.n_set if self.n_set is not None else self.n
        if n < 2: return 0
        else: return self.m2 / (self.n - 1)


def standardize_features(*MS, all=False, **kwargs):
    "Standardize given modalities by feature or by whole matrix"
    axis = None if all else 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ret = [np.nan_to_num((M - M.mean(axis=axis)) / M.std(axis=axis)) for M in MS]
    return clean_return(ret, **kwargs)


def subsample_features(*MS, num_features, **kwargs):
    "Subsample features in given modalities"
    ret = []
    for M, num in zip(MS, num_features):
        idx = np.random.choice(M.shape[1], num, replace=False)
        ret.append(M[:, idx])
    return clean_return(ret, **kwargs)


def pca_features(*MS, num_features, copy=True, **kwargs):
    "Compute PCA on features of given modalities"
    ret = []
    for M, num in zip(MS, num_features):
        pca = sklearn.decomposition.PCA(n_components=num, copy=copy)
        M = pca.fit_transform(M)
        ret.append(M)
    return clean_return(ret, **kwargs)


def subsample_nodes(*DS, num_nodes, **kwargs):
    "Subsample nodes of given arrays"
    idx = np.random.choice(DS[0].shape[0], num_nodes, replace=False)
    ret = [D[idx] for D in DS]
    return clean_return(ret, **kwargs)


class Preprocessing:
    "Apply modifications to input modalities based on given arguments. Takes np.array as input"

    def __init__(
        self,
        # Standardize
        standardize=False,
        # Filtering
        top_variant=None,
        # PCA
        pca_dim=None,
        pca_copy=True,  # Set to false if too much memory being used
        # Subsampling
        num_nodes=None,
        num_features=None,
        # End cast
        device=None,
        **kwargs,
    ):
        self.standardize = standardize
        self.top_variant = top_variant
        self.pca_dim = pca_dim
        self.pca_copy = pca_copy  # Unused if sparse
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.device = device

    
    def fit(self, modalities, **kwargs):
        self.modalities = modalities

        # Standardize
        if self.standardize or self.top_variant is not None:
            self.standardize_mean = [
                np.mean(m, keepdims=True)
                if not scipy.sparse.issparse(m) else
                np.array(np.mean(m).reshape((1, -1)))
                for m in modalities
            ]
            self.standardize_std = [
                np.std(m, keepdims=True)
                if not scipy.sparse.issparse(m) else
                np.array(np.sqrt(m.power(2).mean(axis=0) - np.square(m.mean(axis=0))))
                for m in modalities
            ]

        # Filtering
        if self.top_variant is not None:
            self.filter_mask = [
                np.argsort(std[0])[:-int(var+1):-1]
                if var is not None else None
                for std, var in zip(self.standardize_std, self.top_variant)
            ]
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask)]

        # PCA
        if self.pca_dim is not None:
            self.pca_class = [
                sklearn.decomposition.PCA(
                    n_components=dim,
                    svd_solver='auto' if not scipy.sparse.issparse(m) else 'arpack',
                    copy=self.pca_copy,
                ).fit(m)
                if dim is not None else None
                for m, dim in zip(modalities, self.pca_dim)]

    def transform(self, modalities, **kwargs):
        # Standardize
        # NOTE: Not mean-centered for sparse matrices
        # TODO: Maybe allow for only one dataset to be standardized?
        if self.standardize:
            modalities = [
                (m - m_mean) / np.where(m_std == 0, 1, m_std)
                if not scipy.sparse.issparse(m) else
                (m / np.where(m_std == 0, 1, m_std)).tocsr()
                for m, m_mean, m_std in zip(modalities, self.standardize_mean, self.standardize_std)]

        # Filtering
        if self.top_variant is not None:
            modalities = [m[:, mask] if mask is not None else m for m, mask in zip(modalities, self.filter_mask)]

        # PCA
        if self.pca_dim is not None:
            modalities = [p.transform(m) if p is not None else m for m, p in zip(modalities, self.pca_class)]

        return modalities


    def inverse_transform(self, modalities, **kwargs):
        # PCA
        if self.pca_dim is not None:
            modalities = [p.inverse_transform(m) for m, p in zip(modalities, self.pca_class)]

        # Standardize
        if self.standardize:
            modalities = [
                m_std * m + m_mean
                for m, m_mean, m_std in zip(modalities, self.standardize_mean, self.standardize_std)]

        return modalities

    
    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    

    def cast(self, modalities, copy=True):
        assert self.device is not None, '`device` must be set to call `self.cast`'
        if copy: modalities = modalities.copy()

        # Cast types
        modalities = [torch.tensor(m, dtype=torch.float32, device=self.device) for m in modalities]

        return modalities
    

    def inverse_cast(self, modalities, copy=True):
        if copy: modalities = modalities.copy()

        # Inverse cast
        modalities = [m.detach().cpu().numpy() for m in modalities]

        return modalities
    
    
    def subsample(self, modalities, types=None, partition=None, return_idx=False, **kwargs):
        # Subsample features
        # NOTE: Incompatible with inverse transform
        if self.num_features is not None:
            feature_idx = [np.random.choice(m.shape[1], nf, replace=False) for m, nf in zip(modalities, self.num_features)]
            modalities = [m[:, idx] for m, idx in zip(modalities, feature_idx)]

        node_idx = np.arange(modalities[0].shape[0])
        # Partition
        if partition is not None:
            partition_choice = np.random.choice(np.unique(partition), 1)[0]
            node_idx = node_idx[partition==partition_choice]
        
        # Subsample nodes
        if self.num_nodes is not None:
            assert np.array([m.shape[0] for m in modalities]).var() == 0, 'Nodes in all modalities must be equal to use node subsampling'
            node_idx = np.random.choice(node_idx, self.num_nodes, replace=False)
            
        # Apply subsampling
        modalities = [m[node_idx] for m in modalities]
        if types is not None: types = [t[node_idx] for t in types]

        ret = (modalities,)
        if types is not None: ret += (types,)
        if return_idx: ret += (node_idx,)
        # if self.num_features is not None: ret += (feature_idx,)
        return clean_return(ret)


def overwrite_dict(original, modification, copy=True):
    "Overwrite dictionary values based on provided modification dictionary"

    # Copy if requested
    new = original.copy() if copy else original

    # Write over old keys/Add new keys
    for k, v in modification.items():
        new[k] = v

    return new


class ViewBase:
    def __init__(
        self,
        # Data
        # None
        *args,
        # Data params
        # None
        # Arguments
        ax,
        # Styling
        # None
        **kwargs,
    ):
        # Arguments
        self.ax = ax

        # Initialize plots
        pass

    def update(self, frame):
        # Update plots per frame
        pass


class View3D(ViewBase):
    "Class which controls a plot showing a live 3D view of the environment"
    def __init__(
        self,
        # Data
        states,
        present,
        rewards,
        labels,
        *args,
        # Data params
        dim,
        modal_dist,
        # Arguments
        interval,  # Time between frames
        skip=1,
        # Styling
        num_lines=25,  # Number of attraction and repulsion lines
        rotations_per_second=.1,  # Camera azimuthal rotations per second
        arrow_length_scale=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Storage
        self.states = states
        self.present = present
        self.rewards = rewards
        self.labels = labels
        self.modal_dist = modal_dist

        # Arguments
        self.skip = skip

        # Styling
        self.arrow_length_scale = arrow_length_scale

        # Initialize nodes
        self.get_node_data = lambda frame: states[frame, :, :3]
        self.nodes = [
            self.ax.plot(
                # *get_node_data(0)[labels==l].T,
                [], [],
                label=l,
                linestyle='',
                marker='o',
                ms=6,
                zorder=2.3,
            )[0]
            for l in np.unique(labels)
        ]

        # Initialize velocity arrows
        self.get_arrow_xyz_uvw = lambda frame: (states[frame, :, :3], states[frame, :, dim:dim+3])
        self.arrows = self.ax.quiver(
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
        self.get_distance_discrepancy = lambda frame: [np.array([((states[frame, j, :3] - states[frame, k, :3]).square().sum().sqrt() - dist[j, k].cpu()).item() for j, k in product(*[range(s) for s in dist.shape]) if j < k]) for dist in modal_dist]
        self.get_modal_lines_segments = lambda frame, dist: np.array(states[frame, [[j, k] for j, k in product(*[range(s) for s in dist.shape]) if j < k], :3])
        self.clip_dd_to_alpha = lambda dd: np.clip(np.abs(dd), 0, 2) / 2
        # Randomly select lines to show
        self.line_indices = [[j, k] for j, k in product(*[range(s) for s in modal_dist[0].shape]) if j < k]
        total_lines = int((modal_dist[0].shape[0]**2 - modal_dist[0].shape[0]) / 2)  # Only considers first modality
        self.line_selection = [
            np.random.choice(total_lines, num_lines, replace=False) if num_lines is not None else list(range(total_lines)) for dist in modal_dist
        ]
        self.modal_lines = [
            mp3d.art3d.Line3DCollection(
                self.get_modal_lines_segments(0, dist)[self.line_selection[i]],
                label=f'Modality {i}',
                lw=2,
                zorder=2.1,
            )
            for i, dist in enumerate(modal_dist)
        ]
        for ml in self.modal_lines: self.ax.add_collection(ml)

        # Limits
        self.ax.set(
            xlim=(states[present][:, 0].min(), states[present][:, 0].max()),
            ylim=(states[present][:, 1].min(), states[present][:, 1].max()),
            zlim=(states[present][:, 2].min(), states[present][:, 2].max()),
        )

        # Legends
        l1 = self.ax.legend(handles=self.nodes, loc='upper left')
        self.ax.add_artist(l1)
        l2 = self.ax.legend(handles=[
            self.ax.plot([], [], color='red', label='Repulsion')[0],
            self.ax.plot([], [], color='blue', label='Attraction')[0],
        ], loc='upper right')
        self.ax.add_artist(l2)

        # Styling
        self.ax.set(xlabel='x', ylabel='y', zlabel='z')
        self.get_angle = lambda frame: (30, (360*rotations_per_second)*(frame*interval/1000)-60, 0)
        self.ax.view_init(*self.get_angle(0))

    def update(self, frame):
        super().update(frame)

        # Adjust nodes
        for i, l in enumerate(np.unique(self.labels)):
            present_labels = self.present[frame] * torch.tensor(self.labels==l)
            data = self.get_node_data(frame)[present_labels].T
            self.nodes[i].set_data(*data[:2])
            self.nodes[i].set_3d_properties(data[2])

        # Adjust arrows
        xyz_xyz = [[xyz, xyz+self.arrow_length_scale*uvw] for i, (xyz, uvw) in enumerate(zip(*self.get_arrow_xyz_uvw(frame))) if self.present[frame, i]]
        self.arrows.set_segments(xyz_xyz)

        # Adjust lines
        for i, (dist, ml) in enumerate(zip(self.modal_dist, self.modal_lines)):
            ml.set_segments(self.get_modal_lines_segments(frame, dist)[self.line_selection[i]])
            distance_discrepancy = self.get_distance_discrepancy(frame)[i][self.line_selection[i]]
            color = np.array([(0., 0., 1.) if dd > 0 else (1., 0., 0.) for dd in distance_discrepancy])
            alpha = np.expand_dims(self.clip_dd_to_alpha(distance_discrepancy), -1)
            for j, line_index in enumerate(self.line_selection[i]):
                if not self.present[frame, self.line_indices[line_index]].all(): alpha[j] = 0.
            ml.set_color(np.concatenate((color, alpha), axis=-1))

        # Styling
        self.ax.set_title(f'{self.skip*frame: 4} : {self.rewards[frame].mean():5.2f}')  
        self.ax.view_init(*self.get_angle(frame))


class ViewSilhouette(ViewBase):
    def __init__(
        self,
        # Data
        states,
        labels,
        *args,
        # Data params
        dim,
        # Arguments
        # None
        # Styling
        # None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Data
        self.states = states
        self.labels = labels

        # TODO: Update from 3 to env.dim
        self.get_silhouette_samples = lambda frame: sklearn.metrics.silhouette_samples(self.states[frame, :, :dim].cpu(), self.labels)
        self.bars = [self.ax.bar(l, 0) for l in np.unique(self.labels)]

        # Styling
        self.ax.axhline(y=0, color='black')
        self.ax.set(ylim=(-1, 1))
        self.ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    
    def update(self, frame):
        super().update(frame)

        # Update barplots
        for bar, l in zip(self.bars, np.unique(self.labels)):
            bar[0].set_height(self.get_silhouette_samples(frame)[self.labels==l].mean())

        # Styling
        self.ax.set_title(f'Silhouette Coefficient : {self.get_silhouette_samples(frame).mean():5.2f}') 


# class ViewTemporalBase(ViewBase):
#     def __init__(
#         self,
#         # Data
#         states,
#         present,
#         modalities,
#         *args,
#         # Data params
#         temporal_stages,
#         env,
#         # Arguments
#         # None
#         # Styling
#         # None
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
        
#         # Data
#         self.states = states
#         self.present = present
#         self.modalities = modalities

#         # Data params
#         self.temporal_stages = temporal_stages

#         # Persistent vars
#         self.current_stage = 0
#         self.modal_dist = None

#         # Get temporal discrepancy
#         def get_temporal_discrepancies(frame, recalculate=True):
#             if recalculate: env.set_modalities([m[self.present[frame], :] for m in self.modalities])
#             env.set_positions(states[frame, self.present[frame], :env.dim].to(env.device))
#             return env.get_distance_match().detach().cpu()
#         self.get_temporal_discrepancies = get_temporal_discrepancies

#     def update(self, frame):
#         super().update(frame)

#         # Calculate discrepancy using env
#         recalculate = (frame == 0) or not (self.present[frame] == self.present[frame-1]).all()  # Only recalculate dist if needed
#         discrepancy = self.get_temporal_discrepancy(frame, recalculate=recalculate)

#         # Iterate stage
#         if not ((frame == 0 and len(xdata) > 0)):
#             if frame == 0: self.current_stage = 0
#             if recalculate:
#                 xdata = np.append(xdata, self.current_stage)
#                 ydata = np.append(ydata, None)
#                 self.current_stage += 1  # Technically one ahead
#             ydata[-1] = discrepancy
#             self.temporal_eval_plot.set_xdata(xdata)
#             self.temporal_eval_plot.set_ydata(ydata)


class ViewTemporalDiscrepancy(ViewBase):
    def __init__(
        self,
        # Data
        states,
        present,
        modalities,
        *args,
        # Data params
        temporal_stages,
        env,
        # Arguments
        # None
        # Styling
        # None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Data
        self.states = states
        self.present = present
        self.modalities = modalities

        # Data params
        self.temporal_stages = temporal_stages

        # Persistent vars
        self.current_stage = 0
        self.modal_dist = None

        # Get temporal discrepancy
        def get_temporal_discrepancy(frame, recalculate=True):
            if recalculate: env.set_modalities([m[self.present[frame], :] for m in self.modalities])
            env.set_positions(states[frame, self.present[frame], :env.dim].to(env.device))
            return float(env.get_distance_match().mean().detach().cpu())
        self.get_temporal_discrepancy = get_temporal_discrepancy

        # Initialize plot
        self.temporal_eval_plot = self.ax.plot([], [], color='black', marker='o')[0]
        # TODO: Highlight training regions
 
        # Styling
        self.ax.set_xticks(np.arange(len(self.temporal_stages)), self.temporal_stages)
        self.ax.set_xlim([-.5, len(self.temporal_stages)-.5])
        self.ax.set_ylim([0, 1e0])
        self.ax.set_title('Temporal Discrepancy')
        # ax2.set_yscale('symlog')
        self.ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    def update(self, frame):
        super().update(frame)

        # Calculate discrepancy using env
        recalculate = (frame == 0) or not (self.present[frame] == self.present[frame-1]).all()  # Only recalculate dist if needed
        discrepancy = self.get_temporal_discrepancy(frame, recalculate=recalculate)

        # Adjust plot
        xdata = self.temporal_eval_plot.get_xdata()
        ydata = self.temporal_eval_plot.get_ydata()
        if not ((frame == 0 and len(xdata) > 0)):
            if frame == 0: self.current_stage = 0
            if recalculate:
                xdata = np.append(xdata, self.current_stage)
                ydata = np.append(ydata, None)
                self.current_stage += 1  # Technically one ahead
            ydata[-1] = discrepancy
            self.temporal_eval_plot.set_xdata(xdata)
            self.temporal_eval_plot.set_ydata(ydata)


class ViewTemporalScatter(ViewBase):
    def __init__(
        self,
        # Data
        states,
        present,
        modalities,
        *args,
        # Data params
        env,
        # Arguments
        # None
        # Styling
        # None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Data
        self.states = states
        self.present = present
        self.modalities = modalities

        # Get temporal discrepancies
        def get_temporal_discrepancy(frame, recalculate=True):
            if recalculate: env.set_modalities([m[self.present[frame], :] for m in self.modalities])
            env.set_positions(states[frame, self.present[frame], :env.dim].to(env.device))
            return float(env.get_distance_match().detach().cpu())
        self.get_temporal_discrepancy = get_temporal_discrepancy

        # Initialize plot
        self.points = self.ax.plot(
            [], [],
            linestyle='',
            color='black',
            marker='o',
        )[0]

    def update(self, frame):
        super().update(frame)

        # Get temporal discrepancies
        # self.get_temporal_discrepancy(frame)
        pass

        # Update positions
        # self.points.set_data(*data[:2])
