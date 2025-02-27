from collections import deque
from time import perf_counter
import tracemalloc

import numpy as np


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
