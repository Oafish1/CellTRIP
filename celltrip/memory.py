from collections import defaultdict
import warnings

import numpy as np
import torch

from . import utility as _utility


class AdvancedMemoryBuffer:
    "Memory-efficient implementation of memory"
    def __init__(
        self,
        suffix_len,
        cache_suffix=False,  # Useful if envs and vision are fixed
        rs_nset=1e5,
        split_args={},
        device='cpu',
    ):
        # User parameters
        self.suffix_len = suffix_len
        self.cache_suffix = cache_suffix
        self.rs_nset = rs_nset
        self.split_args = split_args
        self.device = device

        # Storage variables
        self.storage = {
            'keys': [],             # Keys in the first dim of states (1D Tuple)
            'states': [],           # State tensors of dim `keys x non-suffix features` (2D Tensor)
            'actions': [],          # Actions (2D Tensor)
            'action_logs': [],      # Action probabilities (2D Tensor)
            'state_vals': [],       # Critic evaluation of state (1D Tensor)
            'rewards': [],          # Rewards, list of lists (1D List)
            'is_terminals': [],     # Booleans indicating if the terminal state has been reached (Bool)
        }
        self.persistent_storage = {
            'suffixes': {},         # Suffixes corresponding to keys
            'suffix_matrices': {},  # Orderings of suffixes
        }

        # Maintenance variables
        self.recorded = {k: False for k in self.storage}

        # Moving statistics
        self.running_statistics = _utility.train.RunningStatistics(n_set=rs_nset)

    # Sampling
    def __getitem__(self, idx):
        # Parameters
        if not _utility.general.is_list_like(idx): idx = [idx]
        idx = np.array(idx)

        # Initialization
        ret = defaultdict(lambda: [])

        # Sort idx
        sort_idx = np.argsort(idx)
        sort_inverse_idx = np.argsort(sort_idx)

        # Search for index
        current_index = 0
        running_index = 0
        sorted_idx = idx[sort_idx]
        for list_num in range(len(self.storage['keys'])):
            list_len = len(self.storage['keys'][list_num])
            hit_idx = []

            while current_index < len(idx) and running_index + list_len > sorted_idx[current_index]:
                hit_idx += [sorted_idx[current_index]]
                current_index += 1
            
            if len(hit_idx) > 0:
                # Useful shortcuts
                hit_idx = np.array(hit_idx).flatten()
                local_idx = hit_idx - running_index

                # Get values
                for k in self.storage:
                    # Skip certain keys
                    if k in ['keys', 'rewards', 'is_terminals']: continue

                    # Special cases
                    if k == 'states':
                        # `_append_suffix` takes most time without caching, then `split_state`
                        val = _utility.processing.split_state(  # TIME BOTTLENECK
                            self._append_suffix(
                                self.storage[k][list_num],
                                keys=self.storage['keys'][list_num]),  # TIME BOTTLENECK
                            idx=local_idx,
                            **self.split_args,
                        )

                    # Main case
                    else:
                        val = self.storage[k][list_num][local_idx]

                    # Record
                    ret[k].append(val)

            # Iterate list start
            running_index += list_len

            # Break if idx retrieved
            if current_index >= len(idx): break

        # Catch if not all found
        else: raise IndexError(f'Index {sorted_idx[current_index]} out of range')

        # Sort to indexing order and stack
        for k in ret:
            if k == 'states':
                ret[k] = [torch.concat([s[i] for s in ret[k]], dim=0)[sort_inverse_idx] for i in range(2)]
            else:
                ret[k] = torch.concat(ret[k], dim=0)[sort_inverse_idx]

        return dict(ret)
    
    def fast_sample(self, num_memories, mask=None, max_samples_per_state=100, clip=True):
        # Initialization
        ret = defaultdict(lambda: [])
        memory_indices = []

        # Get random list order
        list_order = np.arange(len(self.storage['keys']))
        np.random.shuffle(list_order)

        # Search for index
        num_memories_recorded = 0
        for list_num in list_order:
            if mask is not None:
                if not mask[list_num]: continue
            list_len = len(self.storage['keys'][list_num])

            # Choose random samples
            num_new_memories = min(list_len, max_samples_per_state)
            self_idx = np.random.choice(list_len, num_new_memories, replace=False)

            # Get values
            for k in self.storage:
                # Skip certain keys
                if k in ['keys', 'rewards', 'is_terminals']: continue

                # Special cases
                if k == 'states':
                    val = _utility.processing.split_state(
                        self._append_suffix(
                            self.storage[k][list_num],
                            keys=self.storage['keys'][list_num]),
                        idx=self_idx,
                        **self.split_args,
                    )

                # Main case
                else: val = self.storage[k][list_num][self_idx]

                # Record
                ret[k].append(val)

            # Record memory indices and iterate
            memory_indices += [(list_num, i) for i in self_idx]
            num_memories_recorded += num_new_memories

            # Break if enough memories retrieved
            if num_memories_recorded >= num_memories: break

        # Catch if too few
        else: raise IndexError(f'Memories object only contains {len(self)} memories, {num_memories} requested.')

        # Sort to indexing order and stack
        for k in ret:
            if k == 'states':
                # ret[k] = [torch.concat([s[i] for s in ret[k]], dim=0) for i in range(2)]
                ret[k] = self._concat_states(ret[k])
                if clip: ret[k] = [t[:num_memories] for t in ret[k]]
            else:
                ret[k] = torch.concat(ret[k], dim=0)
                if clip: ret[k] = ret[k][:num_memories]
        memory_indices = torch.tensor(self._index_to_flat_index(memory_indices))
        if clip: memory_indices = memory_indices[:num_memories]

        # TODO: Return indices so rewards can be used
        return dict(ret), memory_indices

    # General
    def __len__(self):
        return sum(len(keys) for keys in self.storage['keys'])
    
    def get_steps(self):
        return len(self.storage['keys'])
    
    def get_storage(self):
        return self.storage, self.persistent_storage

    def append_memory(self, *args):
        "Args is either a memory object or storage, (optional) persistent storage"
        # Parse input
        persistent = True
        if isinstance(args[0], type(self)):
            memory_obj = args[0]
            assert (~np.array(list(memory_obj.recorded.values()))).all(), 'Appending memory object must have no pending records'
            storage, persistent_storage = memory_obj.get_storage()
        elif len(args) > 1:
            storage, persistent_storage = args
        elif len(args) == 1:
            storage, persistent_storage = args, None

        # Check that all records are complete
        assert (~np.array(list(self.recorded.values()))).all(), 'Base memory object must have no pending records'

        # Append storage
        for k in self.storage:
            self.storage[k] += storage[k]

        # Append persistent storage
        if persistent_storage is not None:
            for k in self.persistent_storage:
                self.persistent_storage[k].update(persistent_storage[k])

    def record(self, **kwargs):
        "Record passed variables"
        # Check that passed variables haven't been stored yet for this record
        for k in kwargs:
            assert k in self.storage, f'`{k}` not found in memory object'
            assert not self.recorded[k], f'`{k}` has already been recorded for this record'

        # Store new variables
        for k, v in kwargs.items():
            # Special cases
            if k == 'keys': v = tuple(_utility.general.gen_tolist(v))
            # Cast to device
            else:
                try: v = _utility.processing.recursive_tensor_func(v, lambda x: x.detach().to(self.device))
                except: pass
            # Record
            self.storage[k].append(v)
            self.recorded[k] = True

        # Reset if all variables have been recorded
        if np.array([v for _, v in self.recorded.items()]).all():
            # Gleam suffixes
            keys = self.storage['keys'][-1]
            for j, k in enumerate(keys):
                if k not in self.persistent_storage['suffixes']:
                    self.persistent_storage['suffixes'][k] = self.storage['states'][-1][j][-self.suffix_len:].clone()
                # Check that keys accurately map to suffixes
                # NOTE: Slows down memory appending a small amount
                else:
                    sto_dev = self.storage['states'][-1][j][-self.suffix_len:].device
                    if str(sto_dev) != 'cpu': print(sto_dev)
                    assert (self.storage['states'][-1][j][-self.suffix_len:] == self.persistent_storage['suffixes'][k]).all(), (
                    f'Key `{k}` does not obey 1-to-1 mapping with suffixes')

            # Cut suffixes
            # Note: MUST BE CLONED otherwise stores whole unsliced tensor
            self.storage['states'][-1] = self.storage['states'][-1][..., :-self.suffix_len].clone()

            # Set all variables as unrecorded
            for k in self.recorded: self.recorded[k] = False

    def propagate_rewards(self, gamma=.95, prune=None):
        "Propagate rewards with decay"
        ret, ret_prune, ret_list_prune = [], [], []
        running_rewards = defaultdict(lambda: 0)  # {k: 0 for k in np.unique(sum(self.storage['keys'], ()))}
        running_prune = defaultdict(lambda: 0)  # {k: 0 for k in np.unique(sum(self.storage['keys'], ()))}
        running_list_prune = 0
        for keys, rewards, is_terminal in zip(self.storage['keys'][::-1], self.storage['rewards'][::-1], self.storage['is_terminals'][::-1]):
            # Maybe use as numpy?
            rewards = _utility.general.gen_tolist(rewards)
            for key, reward in zip(keys[::-1], rewards[::-1]):
                if is_terminal:
                    running_list_prune = 0
                    running_rewards[key] = 0  # Reset at terminal state
                    if prune is not None: running_prune[key] = 0
                running_rewards[key] = reward + gamma * running_rewards[key]
                ret.append(running_rewards[key])
                if prune is not None:
                    running_prune[key] += 1
                    ret_prune.append(running_prune[key] > prune)
            if prune is not None:
                running_list_prune += 1
                ret_list_prune.append(running_list_prune > prune)
        ret = torch.tensor(ret[::-1], dtype=torch.float32)
        if prune is not None:
            ret_prune = torch.tensor(ret_prune[::-1], dtype=torch.bool)
            ret_list_prune = torch.tensor(ret_list_prune[::-1], dtype=torch.bool)

        # Need to normalize AFTER propagation
        # NOTE: Approximate using rs_nset last rewards
        for i, r in enumerate(ret[-int(self.rs_nset):]):
            # Don't include pruned rewards in update
            if prune is None or ret_prune[i]: self.running_statistics.update(r)
        # Float vs tensor error happens here when there are no applicable memories
        ret = (ret - self.running_statistics.mean()) / (torch.sqrt(self.running_statistics.variance() + 1e-8))

        if prune is not None:
            return ret, ret_prune, ret_list_prune
        return ret

    def clear(self, clear_persistent=False):
        "Clear memory"
        for k in self.storage: self.storage[k].clear()
        if clear_persistent:
            for k in self.persistent_storage: self.persistent_storage[k].clear()
    
    # Utility
    def _concat_states(self, states):
        # Pad with duplicate nodes when not sufficient
        # NOTE: Inefficient, nested tensor doesn't have enough
        # functionality yet
        shapes = [s[1].shape[1] for s in states]
        max_nodes = max(shapes)
        states = [
            torch.concat([
                s[i] if i == 0 else
                s[i].repeat(
                    1, int(np.ceil(max_nodes/s[i].shape[1])), 1)[:, :max_nodes]
                for s in states],
            dim=0) for i in range(2)]
        return states

    def _append_suffix(self, state, *, keys, cache=False):
        "Append suffixes to state vector with optional cache for common key layouts"
        # Read from cache
        # NOTE: Strings from numpy arrays are slower as keys
        if self.cache_suffix and keys in self.persistent_storage['suffix_matrices']:
            suffix_matrix = self.persistent_storage['suffix_matrices'][keys]

        else:
            # Aggregate suffixes
            suffix_matrix = None
            for k in keys:
                val = self.persistent_storage['suffixes'][k].unsqueeze(0)
                if suffix_matrix is None: suffix_matrix = val
                else: suffix_matrix = torch.concat((suffix_matrix, val), dim=0)

            # Add to cache
            if self.cache_suffix:
                self.persistent_storage['suffix_matrices'][keys] = suffix_matrix
                if len(self.persistent_storage['suffix_matrices']) == 101:
                    warnings.warn(
                        'Persistent storage cache has exceeded 100 entries,'
                        'please verify that there are over 100 unique environment '
                        'states in your data.', RuntimeWarning)

        # Append to state
        return torch.concat((state, suffix_matrix), dim=1)

    def _flat_index_to_index(self, idx, inverse=False):
        """
        Convert int index to grouped format that can be used on keys, state, etc. or vice-versa

        Former takes int or list of ints as input, latter takes tuple or list of tuples
        """
        # Basic checks
        if not inverse and not _utility.general.is_list_like(idx): idx = [idx]
        if inverse and not _utility.general.is_list_like(idx[0]): idx = [idx]

        # Sort idx
        sort_idx = np.argsort(idx, axis=0)
        if inverse: sort_idx = sort_idx[:, 0]
        sort_inverse_idx = np.argsort(sort_idx)

        # Search for index
        seeking_index = 0; running_index = 0; found_idx = []
        for list_num in range(len(self.storage['keys'])):
            list_len = len(self.storage['keys'][list_num])
            if not inverse:
                while seeking_index < len(idx) and idx[sort_idx[seeking_index]] < running_index + list_len:
                    found_idx.append( (list_num, idx[sort_idx[seeking_index]] - running_index) )
                    seeking_index += 1
            else:
                while seeking_index < len(idx) and list_num == idx[sort_idx[seeking_index]][0]:
                    found_idx.append( running_index + idx[sort_idx[seeking_index]][1] )
                    seeking_index += 1
            running_index += list_len

            # Check for exit
            if seeking_index == len(idx): break

        # Throw error if not found
        else: raise IndexError('Index out of range')

        # Invert sorting
        ret = [found_idx[i] for i in sort_inverse_idx]
        if len(ret) == 1: ret = ret[0]
        return ret
    
    def _index_to_flat_index(self, idx):
        return self._flat_index_to_index(idx, inverse=True)
