from collections import defaultdict
import warnings

import numpy as np
import torch

from . import decorator as _decorator
from . import utility as _utility


class AdvancedMemoryBuffer:
    "Memory-efficient implementation of memory"
    def __init__(
        self,
        suffix_len,
        max_memories=int(2*2**30*8/32/32),  # ~2Gb at 16 dimensions
        cache_suffix=True,  # Fastest if True, but should be cleared regularly
        # Propagate
        gae_lambda=.99,
        gae_gamma=.95,
        # prune=0,  # Used to be 100 with non-terminal episodes
        # Sampling
        replay_frac=.25,
        max_samples_per_state=100,
        shuffle=False,
        # Split
        split_args={},
        device='cpu',
    ):
        # User parameters
        self.suffix_len = suffix_len
        self.max_memories = max_memories
        self.cache_suffix = cache_suffix
        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma
        # self.prune = prune
        self.replay_frac = replay_frac
        self.max_samples_per_state = max_samples_per_state
        self.shuffle = shuffle
        self.split_args = split_args
        self.device = device

        # Storage variables
        storage_keys = (
            'keys',                 # Keys in the first dim of states (1D Tuple)
            'states',               # State tensors of dim `keys x non-suffix features` (2D Tensor)
            'actions',              # Actions (2D Tensor)
            'action_logs',          # Action probabilities (2D Tensor)
            'state_vals',           # Critic evaluation of state (1D Tensor)
            'rewards',              # Rewards, list of lists (1D List)
            'is_terminals')         # Booleans indicating if the terminal state has been reached (Bool)
        storage_outcomes = (
            'advantages',           # Advantages, list of lists (1D List)
            'new_memories')         # Booleans indicating if the memory is new (Bool)
        self.storage = {k: [] for k in storage_keys+storage_outcomes}
        persistent_storage_keys = (
            'suffixes',             # Suffixes corresponding to keys
            'suffix_matrices')      # Orderings of suffixes
        self.persistent_storage = {k: {} for k in persistent_storage_keys}
        # Cache for variables which require computation
        self.variable_storage = {}

        # Maintenance variables
        self.recorded = {k: False for k in storage_keys}

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
                    if str(sto_dev) != 'cpu': warnings.warn(f'Tensor found on `{sto_dev}` in `AdvancedMemoryBuffer`.')
                    assert np.isclose(self.storage['states'][-1][j][-self.suffix_len:], self.persistent_storage['suffixes'][k]).all(), (
                    f'Key `{k}` does not obey 1-to-1 mapping with suffixes')
                    if len(self.persistent_storage['suffixes']) > 500_000: warnings.warn(
                        'Number of cached keys has surpassed 500,000.')

            # Cut suffixes
            # Note: MUST BE CLONED otherwise stores whole unsliced tensor
            self.storage['states'][-1] = self.storage['states'][-1][..., :-self.suffix_len].clone()

            # Additional entries
            self.storage['advantages'].append(None)
            self.storage['new_memories'].append(True)

            # Set all variables as unrecorded
            for k in self.recorded: self.recorded[k] = False

        # Clear vars
        self.clear_var_cache()

    def append_memory(self, *args):
        "Args is either a memory object or storage, (optional) persistent storage"
        # Parse input
        if isinstance(args[0], type(self)):
            memory_obj = args[0]
            assert (~np.array(list(memory_obj.recorded.values()))).all(), 'Appending memory object must have no pending records'
            storage, persistent_storage = memory_obj.get_storage()
        elif len(args) > 1:
            storage, persistent_storage = args
        elif len(args) == 1:
            storage, = args
            persistent_storage = None

        # Check that all records are complete
        assert (~np.array(list(self.recorded.values()))).all(), 'Base memory object must have no pending records'

        # Append storage
        for k in self.storage:
            self.storage[k] += storage[k]

        # Append persistent storage
        if persistent_storage is not None:
            for k in self.persistent_storage:
                self.persistent_storage[k].update(persistent_storage[k])

        # Clear vars
        self.clear_var_cache()

    def fast_sample(self, num_memories, replay_frac=None, max_samples_per_state=None, shuffle=None):
        # NOTE: Shuffle should only be used when sequential sampling is taking place
        # Parameters
        if replay_frac is None: replay_frac = self.replay_frac
        if max_samples_per_state is None: max_samples_per_state = self.max_samples_per_state
        if shuffle is None: shuffle = self.shuffle
        num_replay_memories = int(replay_frac * num_memories)
        num_new_memories = num_memories - num_replay_memories

        # Adjust proportions if needed
        total_new_memories = self.get_new_len(max_per_step=max_samples_per_state)
        total_replay_memories = self.get_replay_len(max_per_step=max_samples_per_state)
        if total_new_memories+total_replay_memories < num_memories:
            raise RuntimeError(
                f'Only {total_new_memories+total_replay_memories} possible samples'
                f' with current parameters, {num_memories} requested')
        adjusted = False
        if total_new_memories < num_new_memories:
            num_replay_memories += num_new_memories - total_new_memories
            num_new_memories = total_new_memories
            adjusted = True
        elif total_replay_memories < num_replay_memories:
            num_new_memories += num_replay_memories - total_replay_memories
            num_replay_memories = total_replay_memories
            adjusted = True
        if adjusted:
            new_replay_frac = num_replay_memories / (num_replay_memories+num_new_memories)
            warnings.warn(
                f'Current `replay_frac` ({replay_frac:.3f}) infeasible,'
                f' adjusting to {new_replay_frac:.3f}')

        # Initialization
        ret = defaultdict(lambda: [])
        memory_indices = []

        # Get random list order
        list_order = np.arange(len(self.storage['keys']))
        np.random.shuffle(list_order)

        # Search for index
        num_replay_memories_recorded, num_new_memories_recorded = 0, 0
        for list_num in list_order:
            # Filter by passed mask
            # if mask is not None:
            #     if not mask[list_num]: continue

            # Check if should sample
            if self.storage['new_memories'][list_num]:
                working_memories_recorded = num_new_memories_recorded
                working_memories = num_new_memories
            else:
                working_memories_recorded = num_replay_memories_recorded
                working_memories = num_replay_memories
            if working_memories_recorded >= working_memories: continue

            # Choose random samples
            list_len = len(self.storage['keys'][list_num])
            num_memories_to_add = min(list_len, max_samples_per_state)
            self_idx = np.random.choice(list_len, num_memories_to_add, replace=False)
            if working_memories_recorded + num_memories_to_add > working_memories:
                self_idx = self_idx[:(working_memories-working_memories_recorded)]

            # Get values
            for k in self.storage:
                # Skip certain keys
                if k in ('keys', 'rewards', 'is_terminals', 'prunes', 'new_memories', 'propagated_rewards', 'normalized_rewards'): continue

                # Special cases
                if k == 'states':
                    val = _utility.processing.split_state(
                        self._append_suffix(
                            self.storage[k][list_num],
                            keys=self.storage['keys'][list_num]),
                        idx=self_idx,
                        **self.split_args)

                # Main case
                else: 
                    if k in ('propagated_rewards', 'normalized_rewards') and self.storage[k][list_num] is None:
                        raise ValueError('Make sure to run `self.propagate_rewards(); self.normalize_rewards()` before sampling.')
                    if k in ('advantages',) and self.storage[k][list_num] is None:
                        raise ValueError('Make sure to run `self.compute_advantages()` before sampling.')
                    val = self.storage[k][list_num][self_idx]

                # Record
                ret[k].append(val)

            # Record memory indices and iterate
            # memory_indices += [(list_num, i) for i in self_idx]
            if self.storage['new_memories'][list_num]:
                num_new_memories_recorded += num_memories_to_add
            else:
                num_replay_memories_recorded += num_memories_to_add

            # Break if enough memories retrieved
            if num_replay_memories_recorded + num_new_memories_recorded >= num_memories: break

        # Catch if too few
        else: raise IndexError(
            f'Only able to gather {num_replay_memories_recorded + num_new_memories_recorded}'
            f' memories with current parameters, {num_memories} requested.')

        # Stack tensors
        for k in ret:
            if k == 'states': ret[k] = self._concat_states(ret[k])
            else: ret[k] = torch.concat(ret[k], dim=0)
        # memory_indices = torch.tensor(self._index_to_flat_index(memory_indices))

        # Shuffle
        # NOTE: Not reproducible currently, takes maybe .1 seconds for 10k but
        #       is roughly Omega(N^1.1)
        if shuffle:
            perm = torch.randperm(num_memories)
            for k in ret:
                if k == 'states': ret[k] = [s[perm] for s in ret[k]]
                else: ret[k] = ret[k][perm]
            # memory_indices = memory_indices[perm]

        # Return
        return dict(ret)  # memory_indices
        
    def compute_advantages(self):
        # NOTE: Assumes keys stay the same throughout any single eposide, can be adjusted, however
        for i, (rewards, is_terminal, state_vals, advantages) in enumerate(zip(
            self.storage['rewards'][::-1],
            self.storage['is_terminals'][::-1],
            self.storage['state_vals'][::-1],
            self.storage['advantages'][::-1])):
            # Skip if already calculated
            if advantages is not None: continue

            # Reconfigure index
            i = self.get_steps()-i-1

            # Reset values and advantages if terminal
            if is_terminal:
                running_advantages = previous_state_vals = 0

            # Compute advantage
            deltas = rewards + self.gae_gamma * previous_state_vals - state_vals
            running_advantages = deltas + self.gae_gamma * self.gae_lambda * running_advantages

            # Record
            self.storage['advantages'][i] = running_advantages.clone()
        
    # def adjust_rewards(self):
    #     "Adjust replay rewards from previous reward structure to fit with current new"
    #     # Get new memory mean
    #     # TODO: Try with and without
    #     new_mean = torch.concat(
    #         [s for s, n in zip(self.storage['advantages'], self.storage['new_memories']) if n]).mean()
    #     replay_mean = torch.concat(
    #         [s for s, n in zip(self.storage['advantages'], self.storage['new_memories']) if not n]).mean()
    #     diff_mean = new_mean - replay_mean
    #     for rew, new in zip(self.storage['advantages'], self.storage['new_memories']):
    #         if not new: rew += diff_mean

    def __len__(self):
        if 'len' not in self.variable_storage:
            self.variable_storage['len'] = sum(len(keys) for keys in self.storage['keys'])
            # Check that all keys are same len
            k_len = [len(self.storage[k]) for k in self.storage]
            assert np.isclose(k_len, k_len[0]).all(), (
                f'Memory storage out of sync, found storage key lengths {k_len}'
                f' for keys {list(self.storage.keys())}')
        return self.variable_storage['len']
    
    def get_new_len(self, max_per_step=None, invert=False):
        key = ('new_len', max_per_step) if not invert else ('replay_len', max_per_step)
        if key not in self.variable_storage:
            self.variable_storage[key] = 0
            for state, new_memory in zip(self.storage['states'], self.storage['new_memories']):
                if new_memory^invert:
                    step_len = state.shape[0]
                    if max_per_step is not None: step_len = min(max_per_step, step_len)
                    self.variable_storage[key] += step_len
        return self.variable_storage[key]
    
    def get_replay_len(self, **kwargs):
        return self.get_new_len(invert=True, **kwargs)
    
    def get_new_steps(self, invert=False):
        num_steps = np.sum(self.storage['new_memories'])
        if invert: num_steps = self.get_steps() - num_steps
        return num_steps
    
    def get_replay_steps(self, **kwargs):
        return self.get_new_steps(**kwargs, invert=True)
    
    def get_steps(self):
        return len(self.storage['keys'])
    
    def get_storage(self, persistent=True, new=False):
        # Initialization
        ret = ()

        # Storage
        if new:
            storage = {
                k: [v[i] for i in range(len(v)) if self.storage['new_memories'][i]]
                for k, v in self.storage.items()}
            ret += (storage,)
        else:
            ret += (self.storage,)
            
        if persistent:
            ret += (self.persistent_storage,)

        return ret

    def mark_sampled(self):
        "Mark all steps as sampled"
        for i in range(len(self.storage['new_memories'])):
            if self.storage['new_memories'][i]: self.storage['new_memories'][i] = False

    # Cleanup
    def cull_records(self, max_memories=None):
        "Remove replay memories with reservoir sampling"
        # Parameters
        if max_memories is None: max_memories = self.max_memories
        if max_memories is None: return  # Unlimited memories


        # Cull
        num_memories_to_remove = len(self) - max_memories
        num_memories_removed = 0
        new_steps = np.array(self.storage['new_memories'])
        replay_steps = ~new_steps
        steps_to_remove = np.zeros_like(new_steps, dtype=bool)
        while num_memories_to_remove > num_memories_removed:
            # If replay steps remain
            if replay_steps.sum() > 0:
                idx = np.random.choice(
                    np.argwhere(replay_steps).flatten(), 1)[0]
                steps_to_remove[idx] = True
                replay_steps[idx] = False
                num_memories_removed += self.storage['states'][idx].shape[0]

            # If no replay steps are left
            else:
                warnings.warn(
                    'Insufficient replay memories found to cull, removing new memories instead. Make sure to'
                    ' mark memories after sampling or try raising `max_memories`.')
                idx = np.random.choice(
                    np.argwhere(new_steps).flatten(), 1)[0]
                steps_to_remove[idx] = True
                new_steps[idx] = False
                num_memories_removed += self.storage['states'][idx].shape[0]

        # Remove steps
        idx = np.argwhere(steps_to_remove).flatten()[::-1]
        for k in self.storage:
            for i in idx:
                v = self.storage[k].pop(i)
                if k == 'advantages' and v is None:
                    raise ValueError(
                        '`None` value found for `advantages` while culling.'
                        ' Make sure to call `self.compute_advantages()` before cleaning.')

        # Clear computed vars
        if len(idx) > 0: self.clear_var_cache()

    def clean_keys(self):
        "Clean all unused keys from suffixes"
        # Get all unique keys
        unique_keys = list(set().union(*self.storage['keys']))
        stored_keys = list(self.persistent_storage['suffixes'])
        for k in stored_keys:
            if k not in unique_keys:
                self.persistent_storage['suffixes'].pop(k)

    def clear(self, clear_persistent=False):
        "Clear all memory"
        for k in self.storage: self.storage[k].clear()
        if clear_persistent:
            for k in self.persistent_storage: self.persistent_storage[k].clear()

    def clear_suffix_cache(self):
        "Clear cache"
        self.persistent_storage['suffix_matrices'].clear()

    def clear_var_cache(self):
        "Clear computed vars"
        self.variable_storage.clear()

    def cleanup(self):
        "Clean memory"
        self.cull_records()
        # self.clean_records()
        self.clean_keys()
        self.clear_suffix_cache()
    
    # Utility
    def _concat_states(self, states):
        # Pad with duplicate nodes when not sufficient
        # NOTE: Inefficient, nested tensor doesn't have enough
        # functionality yet
        shapes = [s[1].shape[1] for s in states]
        max_nodes = max(shapes)
        states = [
            torch.concat([
                s[i]
                if i == 0 or np.ceil(max_nodes/s[i].shape[1]) == 1 else
                s[i].repeat(  # TODO: Maybe do NAN instead? Make sure policy can handle it
                    1, int(np.ceil(max_nodes/s[i].shape[1])), 1)[:, :max_nodes]
                for s in states],
            dim=0) for i in range(2)]
        return states

    def _append_suffix(self, state, *, keys):
        "Append suffixes to state vector with optional cache for common key layouts"
        # Read from cache
        # NOTE: Strings from numpy arrays are slower as keys
        if self.cache_suffix and keys in self.persistent_storage['suffix_matrices']:
            suffix_matrix = self.persistent_storage['suffix_matrices'][keys]

        else:
            # Aggregate suffixes
            suffix_matrix = []
            for k in keys:
                val = self.persistent_storage['suffixes'][k]
                suffix_matrix.append(val)
            suffix_matrix = torch.stack(suffix_matrix, dim=0)

            # Add to cache
            if self.cache_suffix:
                self.persistent_storage['suffix_matrices'][keys] = suffix_matrix
                if len(self.persistent_storage['suffix_matrices']) == 101:
                    warnings.warn(
                        'Persistent storage cache has exceeded 100 entries,'
                        ' please verify that there are over 100 unique environment'
                        ' states in your data.', RuntimeWarning)

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
