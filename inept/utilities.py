from collections import deque
from time import perf_counter
import tracemalloc
import warnings

import numpy as np
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
        method='absolute',
        buffer=50,
        delta=.01,
        decreasing=False,

        # `average` method parameters
        window_size=10,

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


def clean_return(ret):
    "Clean return output for improved parsing"
    if len(ret) == 1: return ret[0]
    return ret


def normalize(*MS):
    "Normalize given modalities by feature"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ret = [np.nan_to_num((M - M.mean(axis=0)) / M.std(axis=0)) for M in MS]
    return clean_return(ret)


def subsample_features(*MS, num_features):
    "Subsample features in given modalities"
    ret = []
    for M, num in zip(MS, num_features):
        idx = np.random.choice(M.shape[1], num, replace=False)
        ret.append(M[:, idx])
    return clean_return(ret)


def pca_features(*MS, num_features):
    "Compute PCA on features of given modalities"
    ret = []
    for M, num in zip(MS, num_features):
        pca = sklearn.decomposition.PCA(num)
        M = pca.fit_transform(M)
        ret.append(M)
    return clean_return(ret)


def subsample_nodes(*DS, num_nodes):
    "Subsample nodes of given arrays"
    idx = np.random.choice(DS[0].shape[0], num_nodes, replace=False)
    ret = [D[idx] for D in DS]
    return clean_return(ret)


def split_state(state, idx=None):
    "Split full state matrix into individual inputs, idx is an optional array"
    # Parameters
    if idx is None: idx = list(range(state.shape[0]))
    if not isinstance(idx, list): idx = [idx]
    # Get self features for each node
    self_entity = state[idx]
    # Get node features for each state
    node_entities = state.unsqueeze(0).expand(len(idx), *state.shape)
    mask = torch.zeros((len(idx), state.shape[0]), dtype=torch.bool)
    for i, j in enumerate(idx): mask[i, j] = True
    mask = ~mask
    node_entities = node_entities[mask].reshape(len(idx), state.shape[0]-1, state.shape[1])
    # Return
    return self_entity, node_entities


def is_list_like(l):
    "Test if `l` has the `__len__` method`"
    try:
        len(l)
        return True
    except:
        return False
