from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from . import decorator as _decorator
from . import utility as _utility


class AdvancedMemoryBuffer:
    "Memory-efficient implementation of memory"
    def __init__(
        self,
        suffix_len,
        max_memories=int(2*2**30*8/32/32),  # ~2Gb at 16 dimensions
        cache_suffix=True,  # Fastest if True, but should be cleared regularly
        # Recording
        flush_on_record=True,
        # Propagate
        gae_lambda=.95,
        gamma=.99,
        # Sampling
        replay_frac=0,
        max_samples_per_state=np.inf,
        use_pre_concat_if_possible=True,  # Uses a bunch more memory, but much faster if choice or uniform sampling
        uniform=False,
        shuffle=False,
        # Culling
        prune=None,
        cull_by_episode=False,
        max_staleness=None,
        # Split
        split_kwargs={},
        device='cpu',
    ):
        # User parameters
        self.suffix_len = suffix_len
        self.max_memories = max_memories
        self.cache_suffix = cache_suffix
        self.flush_on_record = flush_on_record
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.replay_frac = replay_frac
        self.max_samples_per_state = max_samples_per_state
        self.shuffle = shuffle
        self.use_pre_concat_if_possible = use_pre_concat_if_possible
        self.uniform = uniform
        self.prune = prune
        self.cull_by_episode = cull_by_episode
        if max_staleness is None:
            self.max_staleness = torch.inf if replay_frac > 0 else 0
        else: self.max_staleness = max_staleness
        self.split_kwargs = split_kwargs
        self.device = device

        # Storage variables
        storage_keys = (
            'keys',                     # Keys in the first dim of states (1D Tuple)
            'states',                   # State tensors of dim `keys x non-suffix features` (2D Tensor)
            'actions',                  # Actions (2D Tensor)
            'action_logs',              # Action probabilities (2D Tensor)
            'state_vals',               # Critic evaluation of state (1D Tensor)
            'rewards',                  # Rewards, list of lists (1D List)
            'is_truncateds',            # Booleans indicating if the end state has been reached (Bool)
            'is_naturals',               # The end state was reached by env rather than manually (Bool)
            'terminal_states',          # State tensors after terminal action (2D Tensor)
            'terminal_state_vals')      # Critic evaluation of terminal state (1D Tensor)
        storage_outcomes = (
            'advantages',               # Advantages, list of tensors (1D Tensor)
            'propagated_rewards',       # Propagated rewards, list of tensors (1D Tensor)
            'prunes',
            'staleness')                # Int indicating the staleness of the memory (Int)
        self.storage = {k: [] for k in storage_keys+storage_outcomes}
        temporary_keys = (
            'target_modalities',)       # Target modalities corresponding to keys
        self.temporary_storage = {k: None for k in temporary_keys}
        persistent_storage_keys = (
            'suffixes',                  # Suffixes corresponding to keys
            'target_suffixes',           # Target modalities corresponding to keys
            'target_suffix_matrices',  # Target modalities corresponding to keys
            'suffix_matrices')           # Orderings of suffixes
        self.persistent_storage = {k: {} for k in persistent_storage_keys}
        # Cache for variables which require computation
        self.variable_storage = {}
        self.buffer = []

        # Maintenance variables
        self.recorded = {k: False for k in storage_keys}

    def record_buffer(self, **kwargs):
        self.buffer.append(kwargs)
        if self.flush_on_record: self.flush_buffer()

    def flush_buffer(self):
        for kwargs in self.buffer: self.record(**kwargs)
        self.buffer.clear()

    # @_decorator.profile
    # @_decorator.line_profile
    def record(self, **kwargs):
        "Record passed variables"
        # Check that passed variables haven't been stored yet for this record
        for k in kwargs:
            assert k in self.storage or k in self.temporary_storage, f'`{k}` not found in memory object'
            if k in self.storage:
                assert not self.recorded[k], f'`{k}` has already been recorded for this record'

        # Store new variables
        for k, v in kwargs.items():
            # Special cases
            if k == 'keys': v = tuple(_utility.general.gen_tolist(v))
            elif k == 'is_truncateds':
                if not v:
                    for k1 in ('terminal_states', 'terminal_state_vals'):
                        self.storage[k1].append(None)
                        self.recorded[k1] = True
            # Key skimmers
            elif k == 'target_modalities':
                self.temporary_storage['target_modalities'] = v.detach().cpu().numpy()
                continue
            # Cast to device
            else:
                try: v = _utility.processing.recursive_tensor_func(v, lambda x: x.detach().cpu().numpy())
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
                    self.persistent_storage['suffixes'][k] = self.storage['states'][-1][j][-self.suffix_len:].copy()
                if self.temporary_storage['target_modalities'] is not None and k not in self.persistent_storage['target_suffixes']:
                    self.persistent_storage['target_suffixes'][k] = self.temporary_storage['target_modalities'][j].copy()
                # Check that keys accurately map to suffixes
                # NOTE: Slows down memory appending a small amount
                # else:
                    # sto_dev = self.storage['states'][-1][j][-self.suffix_len:].device
                    # if str(sto_dev) != 'cpu': warnings.warn(f'Tensor found on `{sto_dev}` in `AdvancedMemoryBuffer`.')
                    # assert np.isclose(self.storage['states'][-1][j][-self.suffix_len:], self.persistent_storage['suffixes'][k]).all(), (
                    #     f'Key `{k}` does not obey 1-to-1 mapping with suffixes')
                if len(self.persistent_storage['suffixes']) > 500_000: warnings.warn(
                    'Number of cached keys has surpassed 500,000.')
                if len(self.persistent_storage['target_suffixes']) > 500_000: warnings.warn(
                    'Number of cached keys has surpassed 500,000.')

            # Cut suffixes
            # Note: MUST BE CLONED otherwise stores whole unsliced tensor
            self.storage['states'][-1] = self.storage['states'][-1][..., :-self.suffix_len].copy()
            if self.storage['terminal_states'][-1] is not None: self.storage['terminal_states'][-1] = self.storage['terminal_states'][-1][..., :-self.suffix_len].copy()

            # Additional entries
            self.storage['propagated_rewards'].append(None)
            self.storage['advantages'].append(None)
            self.storage['prunes'].append(False)
            self.storage['staleness'].append(0)

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

    def __getitem__(self, idx):
        # Params
        pre_append = True  # Append node features once during concatenation stage then never again
        pre_cast = True  # Cast to tensor once during concatenation stage and never again
        pad_nodes = True  # Pad node array to avoid indexing
        general_storage_keys = ('actions', 'action_logs', 'state_vals', 'advantages', 'propagated_rewards')
        assert self.storage['advantages'][0] is not None, 'Please run `self.compute_advantages` before indexing'
    
        # Pre concatenate vars
        if 'concatenated_buffer' not in self.variable_storage:
            self.variable_storage['concatenated_buffer'] = {}
            # General case
            for k in general_storage_keys:
                self.variable_storage['concatenated_buffer'][k] = np.concat(self.storage[k], axis=0)
            # Get key indices
            self.variable_storage['concatenated_buffer']['keys'] = np.concat(self.storage['keys'])
            unique_keys, new_suffix_idx = np.unique(self.variable_storage['concatenated_buffer']['keys'], return_inverse=True, axis=-1)
            try:
                self.variable_storage['concatenated_buffer']['suffixes'] = np.stack([
                    self.persistent_storage['suffixes'][k] for k in unique_keys], axis=0)
            except:
                print(unique_keys)
                print(self.persistent_storage.keys())
                raise RuntimeError('Something went wrong with suffix concatenation.')
            self.variable_storage['concatenated_buffer']['idx_to_suffix'] = new_suffix_idx
            state_lens = np.array(list(map(len, self.storage['keys'])))
            suffix_keys_padded = -np.ones((state_lens.shape[0], state_lens.max()), dtype=int)
            suffix_keys_padded_mask = np.expand_dims(np.arange(state_lens.max()), axis=0) < np.expand_dims(state_lens, axis=-1)
            suffix_keys_padded[suffix_keys_padded_mask] = new_suffix_idx
            self.variable_storage['concatenated_buffer']['suffix_indices'] = suffix_keys_padded
            # Get state indices
            mask = _utility.general.padded_stack(list(map(lambda x: np.ones_like(x, dtype=bool), self.storage['keys'])), method='false')
            grid = np.meshgrid(*[np.arange(s) for s in mask.shape], indexing='ij')
            self.variable_storage['concatenated_buffer']['idx_to_buffer'] = np.stack([arr[mask] for arr in grid], axis=-1)  # TODO: Try on variable-sized envs, should work
            # Concatenate states
            if pre_append: states = [self._append_suffix(s, keys=k) for k, s in zip(self.storage['keys'], self.storage['states'])]
            else: states = self.storage['states']
            # Handling for modifiable vision states
            # does_split = len(_utility.processing.split_state(states[0], **self.split_kwargs)) > 1
            # if does_split:
            split_states = [_utility.processing.split_state(state, force_split=True, **self.split_kwargs) for state in states]
            s0 = _utility.general.padded_stack([s[0] for s in split_states], method='zero')
            s1 = _utility.general.padded_stack([s[1] for s in split_states], method='zero')
            s2 =  _utility.general.padded_stack([s[2] for s in split_states], method='true')
            # Regular
            # else:
            #     s0 = _utility.general.padded_stack(states, method='zero')
            #     s2 =  _utility.general.padded_stack(list(map(lambda x: np.eye(x.shape[0], dtype=bool), states)), method='true')
            if pre_cast:
                s0 = torch.from_numpy(s0).to(torch.get_default_dtype()).to(self.device)  # Cast to tensor early to avoid repeated casting cost
                # if does_split:
                s1 = torch.from_numpy(s1).to(torch.get_default_dtype()).to(self.device)  # Cast to tensor early to avoid repeated casting cost
                # else: s1 = s0
                s2 = torch.from_numpy(s2).to(torch.bool).to(self.device)
            states = [s0, s1, s2]
            self.variable_storage['concatenated_buffer']['states'] = states
            self.out = None

        # Automatically determine idx if int
        if isinstance(idx, int):
            idx = [idx]
            # TODO: Make clustered version of this
        idx = np.sort(np.array(idx))
        # return self.fast_sample(chosen=idx)

        # Get indices to padded buffer from idx
        buffer_idx = self.variable_storage['concatenated_buffer']['idx_to_buffer'][idx]
        unique_list_nums, unique_list_counts = np.unique(buffer_idx[:, 0], return_counts=True)
        if pad_nodes:
            size = self.variable_storage['concatenated_buffer']['states'][1].shape[0]
            new_unique_list_nums = np.arange(size, dtype=int)
            mask = np.isin(new_unique_list_nums, unique_list_nums)
            new_unique_list_counts = np.zeros(size, dtype=int)
            new_unique_list_counts[mask] = unique_list_counts
            unique_list_nums = new_unique_list_nums
            unique_list_counts = new_unique_list_counts
        grouped_indices = -np.ones((unique_list_nums.shape[0], unique_list_counts.max()), dtype=int)
        grouped_sample_mask = np.expand_dims(np.arange(unique_list_counts.max()), axis=0) < np.expand_dims(unique_list_counts, axis=-1)
        grouped_indices[grouped_sample_mask] = buffer_idx[:, 1]  # np.argsort(unique_list_inverse) doesn't respect initial ordering
        overall_indices = grouped_indices.copy()
        overall_indices[grouped_sample_mask] = idx

        # Format output
        # feature_dim = self.storage['states'][0].shape[-1] + self.suffix_len
        # if self.out is None:
        #     self.out = [
        #         np.zeros((*grouped_indices.shape, feature_dim)),
        #         np.zeros((grouped_indices.shape[0], self.variable_storage['concatenated_buffer']['states'][1].shape[1], feature_dim)),
        #         np.zeros((*grouped_indices.shape, self.variable_storage['concatenated_buffer']['states'][1].shape[1])),
        #     ]
        # states = self.out
        # Format states
        # states[0][..., :-self.suffix_len] = self.variable_storage['concatenated_buffer']['states'][0][np.expand_dims(unique_list_nums, axis=-1), grouped_indices]
        # states[1][..., :-self.suffix_len] = self.variable_storage['concatenated_buffer']['states'][1][unique_list_nums]
        # states[2][:] = self.variable_storage['concatenated_buffer']['states'][2][np.expand_dims(unique_list_nums, axis=-1), grouped_indices]

        # Format states
        states = [
            self.variable_storage['concatenated_buffer']['states'][0][np.expand_dims(unique_list_nums, axis=-1), grouped_indices],
            self.variable_storage['concatenated_buffer']['states'][1][unique_list_nums if not pad_nodes else slice(unique_list_nums.shape[0])],
            self.variable_storage['concatenated_buffer']['states'][2][np.expand_dims(unique_list_nums, axis=-1), grouped_indices]]
        if pre_cast: states[2] = states[2].clone()
        else: states[2] = states[2].copy()
        # Post-append
        if not pre_append:
            suffix_idx = self.variable_storage['concatenated_buffer']['idx_to_suffix'][idx]
            suffix_grouped_indices = grouped_indices.copy()
            suffix_grouped_indices[grouped_sample_mask] = suffix_idx
            # states[0][..., -self.suffix_len:] = self.variable_storage['concatenated_buffer']['suffixes'][suffix_grouped_indices.flatten()].reshape((*suffix_grouped_indices.shape, -1))
            # states[1][..., -self.suffix_len:] = self.variable_storage['concatenated_buffer']['suffixes'][self.variable_storage['concatenated_buffer']['suffix_indices'][unique_list_nums]]
            # s0_suffix = self.variable_storage['concatenated_buffer']['suffixes'][suffix_grouped_indices.flatten()].reshape((*suffix_grouped_indices.shape, -1))
            # s1_suffix = self.variable_storage['concatenated_buffer']['suffixes'][self.variable_storage['concatenated_buffer']['suffix_indices'][unique_list_nums]]
            # states[0] = np.concatenate((states[0], s0_suffix), axis=-1)
            # states[1] = np.concatenate((states[1], s1_suffix), axis=-1)
        # Mask ragged fillers, uses -1 as indicator
        states[2][tuple(np.argwhere(grouped_indices==-1).T)] = True

        # Pull from padded buffer and cast to tensors
        ret = {}
        # General case
        for k in general_storage_keys:
            ret[k] = torch.from_numpy(self.variable_storage['concatenated_buffer'][k][idx]).to(torch.get_default_dtype()).to(self.device)
        # State case
        if pre_cast: ret['states'] = states
        else:
            ret['states'] = [
                torch.from_numpy(states[0]).to(torch.get_default_dtype()).to(self.device),  # Source states (All timesteps x Nodes x Features)
                torch.from_numpy(states[1]).to(torch.get_default_dtype()).to(self.device),  # Visible states (All timesteps x Visible nodes x Features)
                torch.from_numpy(states[2]).to(torch.bool).to(self.device)]  # Attention mask (All timesteps x Nodes x Visible nodes)
        return overall_indices, ret  # indices, data
    
    def get_terminal_pairs(self):
        # Get eligible states
        terminal_state_idx = np.argwhere(self.storage['is_naturals']).flatten()
        if len(terminal_state_idx) == 0: return None, None
        keys, states = [], []
        for i in terminal_state_idx:
            keys.append(self.storage['keys'][i])
            states.append(self.storage['states'][i])  # 'terminal_states' would be more correct, but 'states' used for more generality because it doesn't matter that much
        
        # Format state returns
        # does_split = len(_utility.processing.split_state(states[0], **self.split_kwargs)) > 1
        # Splittable
        # if does_split:
        split_states = [_utility.processing.split_state(state, force_split=True, **self.split_kwargs) for state in states]
        s0 = _utility.general.padded_stack([s[0] for s in split_states], method='zero')
        s1 = _utility.general.padded_stack([s[1] for s in split_states], method='zero')
        s2 = _utility.general.padded_stack([s[2] for s in split_states], method='true')
        s0 = torch.from_numpy(s0).to(torch.get_default_dtype()).to(self.device)
        s1 = torch.from_numpy(s1).to(torch.get_default_dtype()).to(self.device)
        s2 = torch.from_numpy(s2).to(torch.bool).to(self.device)
        states = [s0, s1, s2]
        # Regular
        # else:
        #     s01 = _utility.general.padded_stack(states, method='zero')
        #     s2 =  _utility.general.padded_stack(list(map(lambda x: np.eye(x.shape[0], dtype=bool), states)), method='true')
        #     s01 = torch.from_numpy(s01).to(torch.get_default_dtype()).to(self.device)
        #     s2 = torch.from_numpy(s2).to(torch.bool).to(self.device)
        #     states = [s01, s01, s2]

        # Format data returns
        target_modalities = _utility.general.padded_stack([
            self._append_suffix(
                keys=k,
                suffix_key='target_suffixes',
                suffix_matrix_key='target_suffix_matrices')
            for k in keys], method='zero')
        target_modalities = torch.from_numpy(target_modalities).to(torch.get_default_dtype()).to(self.device)
        
        return states, target_modalities

    # def fast_sample(
    #     self, num_memories=None, chosen=None, replay_frac=None, max_samples_per_state=None,
    #     double_batch=True, flatten=False, use_pre_concat_if_possible=None, pad_indexer=True,
    #     uniform=None, shuffle=None, round_sample=None):
    #     # NOTE: Shuffle should only be used when sequential sampling is taking place
    #     # NOTE: Chosen returns sorted idx, flatten with double batch will duplicate node info for each sample
    #     # Assertions
    #     assert num_memories is not None or chosen is not None

    #     # Parameters
    #     if replay_frac is None: replay_frac = self.replay_frac
    #     if max_samples_per_state is None: max_samples_per_state = self.max_samples_per_state
    #     if use_pre_concat_if_possible is None: use_pre_concat_if_possible = self.use_pre_concat_if_possible
    #     if uniform is None: uniform = self.uniform
    #     if shuffle is None: shuffle = self.shuffle

    #     # Preemptively concatenate and pad
    #     pre_concat = use_pre_concat_if_possible
    #     if pre_concat and 'concatenated_states' not in self.variable_storage:
    #         # NOTE: Technically could be compatible with no double_batch, but would be insanely large
    #         concatenated_states = self._concat_states(
    #             [(self._append_suffix(s, keys=k),) for k, s in zip(self.storage['keys'], self.storage['states'])],
    #             double_batch=double_batch)
    #         self.variable_storage['concatenated_states'] = concatenated_states

    #     # Adjust proportions if needed
    #     if num_memories is not None:
    #         # Set preliminary vars
    #         num_replay_memories = int(replay_frac * num_memories)
    #         num_new_memories = num_memories - num_replay_memories

    #         # Calculate memory balance
    #         total_new_memories = self.get_new_len(max_per_step=max_samples_per_state)
    #         total_replay_memories = self.get_replay_len(max_per_step=max_samples_per_state)
    #         if total_new_memories+total_replay_memories < num_memories:
    #             raise RuntimeError(
    #                 f'Only {total_new_memories+total_replay_memories} possible samples'
    #                 f' with current parameters, {num_memories} requested')
    #         adjusted = False
    #         if total_new_memories < num_new_memories:
    #             num_replay_memories += num_new_memories - total_new_memories
    #             num_new_memories = total_new_memories
    #             adjusted = True
    #         elif total_replay_memories < num_replay_memories:
    #             num_new_memories += num_replay_memories - total_replay_memories
    #             num_replay_memories = total_replay_memories
    #             adjusted = True
    #         if adjusted:
    #             new_replay_frac = num_replay_memories / (num_replay_memories+num_new_memories)
    #             warnings.warn(
    #                 f'Current `replay_frac` ({replay_frac:.3f}) infeasible,'
    #                 f' adjusting to {new_replay_frac:.3f}')

    #     # Initialization
    #     ret = defaultdict(lambda: [])

    #     # List order
    #     list_order = np.arange(len(self.storage['keys']))
    #     if chosen is not None:
    #         # Chosen idx
    #         memories_to_record = np.array(chosen)
    #         if num_memories is None: num_memories = memories_to_record.shape[0]
    #     elif uniform:
    #         # Uniform sampling (could also work with duplicates)
    #         new_memories_to_record = np.random.choice(self.get_new_len(), num_new_memories, replace=False)
    #         replay_memories_to_record = np.random.choice(self.get_replay_len(), num_replay_memories, replace=False)
    #     else:
    #         # Get random list order
    #         np.random.shuffle(list_order)

    #     # Search for index
    #     num_replay_memories_recorded, num_new_memories_recorded = np.array(0), np.array(0)
    #     for list_num in list_order:
    #         # Check if should sample
    #         if chosen is not None:
    #             working_memories_recorded = num_new_memories_recorded
    #             working_memories = num_memories
    #             working_memories_to_record = memories_to_record
    #         elif self.storage['staleness'][list_num] == 0:
    #             working_memories_recorded = num_new_memories_recorded
    #             working_memories = num_new_memories
    #             if uniform: working_memories_to_record = new_memories_to_record
    #         else:
    #             working_memories_recorded = num_replay_memories_recorded
    #             working_memories = num_replay_memories
    #             if uniform: working_memories_to_record = replay_memories_to_record
    #         if working_memories_recorded >= working_memories: continue

    #         # Choose random samples
    #         list_len = len(self.storage['keys'][list_num])
    #         if uniform or chosen is not None:
    #             # Uniformly add
    #             mask = working_memories_to_record < list_len
    #             num_memories_to_add = mask.sum()
    #             if num_memories_to_add == 0:
    #                 working_memories_to_record -= list_len
    #                 continue
    #             self_idx = working_memories_to_record[mask].copy()
    #             working_memories_to_record[mask] = len(self)  # Kinda hacky, but works
    #             working_memories_to_record -= list_len
    #         else:
    #             # Greedily add
    #             num_memories_to_add = min(list_len, max_samples_per_state)
    #             if working_memories_recorded + num_memories_to_add > working_memories and round_sample != 'up':
    #                 if round_sample is None:
    #                     num_memories_to_add = working_memories-working_memories_recorded
    #                 elif round_sample == 'down': break
    #                 else: raise RuntimeError(f'`round_sample` method `{round_sample}` not implemented.')
    #             if list_len != num_memories_to_add:
    #                 if round_sample is None:
    #                     self_idx = np.random.choice(list_len, num_memories_to_add, replace=False)
    #             else: self_idx = np.arange(list_len)

    #         # Get values
    #         for k in self.storage:
    #             # Skip certain keys
    #             if k not in ('states', 'actions', 'action_logs', 'state_vals', 'advantages', 'propagated_rewards', 'staleness'): continue

    #             # Special cases
    #             if k == 'states':
    #                 if pre_concat:
    #                     # val = list_num
    #                     val = (list_num, list(self_idx))
    #                 else:
    #                     val = _utility.processing.split_state(
    #                         self._append_suffix(
    #                             self.storage[k][list_num],
    #                             keys=self.storage['keys'][list_num]),
    #                         idx=self_idx,
    #                         **self.split_kwargs)
                    
    #             # Single value case
    #             elif k in ('staleness',):
    #                 val = np.array(len(self_idx)*[self.storage[k][list_num]])

    #             # Main case
    #             else: 
    #                 if k in ('propagated_rewards', 'normalized_rewards') and self.storage[k][list_num] is None:
    #                     raise ValueError('Make sure to run `self.propagate_rewards(); self.normalize_rewards()` before sampling.')
    #                 if k in ('advantages',) and self.storage[k][list_num] is None:
    #                     raise ValueError('Make sure to run `self.compute_advantages()` before sampling.')
    #                 if (len(self_idx) == self.storage[k][list_num].shape[0]) and (self_idx[:-1] < self_idx[1:]).all():
    #                     val = self.storage[k][list_num]
    #                 else:
    #                     val = self.storage[k][list_num][self_idx]

    #             # Record
    #             ret[k].append(val)

    #         # Record memory indices and iterate
    #         # memory_indices += [(list_num, i) for i in self_idx]
    #         if self.storage['staleness'][list_num] == 0:
    #             num_new_memories_recorded += num_memories_to_add
    #         else:
    #             num_replay_memories_recorded += num_memories_to_add

    #         # Break if enough memories retrieved
    #         if num_replay_memories_recorded + num_new_memories_recorded >= num_memories: break

    #     # Catch if too few
    #     else: warnings.warn(
    #         f'Only able to gather {num_replay_memories_recorded + num_new_memories_recorded}'
    #         f' memories with current parameters, {num_memories} requested.')

    #     # Stack tensors
    #     for k in ret:
    #         if k == 'states':
    #             if pre_concat:
    #                 # Non-flexible
    #                 # ret[k] = _utility.processing.recursive_tensor_func(
    #                 #     self.variable_storage['concatenated_states'], lambda t: t[ret[k]])

    #                 # Individually indexable and masking
    #                 # c = [(0, [10, 20]), (1, [50, 60, 70]), (3, [5])]
    #                 if flatten: ret[k] = [(ln, [si]) for ln, sis in ret[k] for si in sis]
    #                 elif pad_indexer:
    #                     # Pad idx for 0 and 2 out_states so that 1 may not be indexed, saving time in the noted area
    #                     last_num = self.variable_storage['concatenated_states'][1].shape[0]
    #                     for i in range(len(ret[k])-1, -1, -1):
    #                         while ret[k][i][0] < last_num-1:
    #                             ret[k].insert(i+1, (last_num-1, [-1]))
    #                             last_num -= 1
    #                         last_num = ret[k][i][0]
    #                     # Handle unsampled 0 and sequential indices
    #                     while last_num > 0:
    #                         ret[k].insert(0, (last_num-1, [-1]))
    #                         last_num -= 1

    #                 indexing_arr = [np.array([list(map(lambda x: x[0], ret[k]))], dtype=np.long).T]
    #                 if pad_indexer:
    #                     # Assert in order, unique, and full
    #                     assert indexing_arr[0].shape[0] == self.variable_storage['concatenated_states'][1].shape[0]
    #                     assert np.all(indexing_arr[0].flatten()[1:] > indexing_arr[0].flatten()[:-1])
    #                 ragged_arr = list(map(lambda x: x[1], ret[k]))
    #                 max_len = max(len(l) for l in ragged_arr)
    #                 indexing_arr.append(np.array([l+(max_len-len(l))*[-1] for l in ragged_arr], dtype=np.long))
    #                 # indexing_arr[0] = np.broadcast_to(indexing_arr[0], indexing_arr[1].shape)
    #                 # slicify = True  # len(ret[k]) > .9 * self.variable_storage['concatenated_states'][1].shape[0]
    #                 # if slicify:
    #                 #     # Convert to slices to avoid noted slowdown in the nearly full case
    #                 #     sliced_indexing_arr_0 = _utility.general.slicify_array(indexing_arr[0].squeeze(-1))
    #                 #     print(sliced_indexing_arr_0)
    #                 out_states = (
    #                     self.variable_storage['concatenated_states'][0][indexing_arr[0], indexing_arr[1]],
    #                     self.variable_storage['concatenated_states'][1][indexing_arr[0].squeeze(-1)]  # SLOWDOWN IF INDEXED
    #                     if not pad_indexer or flatten else self.variable_storage['concatenated_states'][1],
    #                     # if not slicify else torch.concat([self.variable_storage['concatenated_states'][1][sl] for sl in sliced_indexing_arr_0], dim=0),
    #                     self.variable_storage['concatenated_states'][2][indexing_arr[0], indexing_arr[1]].clone())
    #                 out_states[2][tuple(np.argwhere(indexing_arr[1]==-1).T)] = True  # Mask ragged fillers, uses -1 as indicator
    #                 ret[k] = out_states
    #                 if flatten: ret[k] = [rk.squeeze(1) if i!=1 else rk for i, rk in enumerate(ret[k])]

    #             else: ret[k] = self._concat_states(ret[k], double_batch=double_batch)

    #         else: ret[k] = torch.from_numpy(np.concat(ret[k], axis=0)).to(torch.get_default_dtype()).to(self.device)
    #     # memory_indices = torch.tensor(self._index_to_flat_index(memory_indices))

    #     # Shuffle
    #     # NOTE: Not reproducible currently, takes maybe .1 seconds for 10k but
    #     #       is roughly Omega(N^1.1)
    #     if shuffle:
    #         perm = torch.randperm(num_memories)
    #         for k in ret:
    #             if k == 'states': ret[k] = [s[perm] for s in ret[k]]
    #             else: ret[k] = ret[k][perm]
    #         # memory_indices = memory_indices[perm]

    #     # Return
    #     return dict(ret)  # memory_indices
    
    def feed_new(self, moving_class, key='rewards'):
        # NOTE: Should update AFTER computing advantages, as otherwise rewards and state vals are not on the same policy (Maybe?)
        moving_class.update(torch.cat([t for t, stale in zip(self.storage[key], self.storage['staleness']) if stale==0]).to(moving_class.mean.device))
        
    def compute_advantages(self, gamma=None, gae_lambda=None, prune=None, normalize_rewards=False, moving_standardization=None):
        # NOTE: Assumes keys stay the same throughout any single eposide, can be adjusted, however
        # Default values
        if gamma is None: gamma = self.gamma
        if gae_lambda is None: gae_lambda = self.gae_lambda
        if prune is None: prune = self.prune

        # Normalize
        if normalize_rewards:
            # rewards_mean = torch.cat([rew for rew, stale in zip(self.storage['rewards'], self.storage['staleness']) if stale==0]).mean()
            # rewards_std = torch.cat([rew for rew, stale in zip(self.storage['rewards'], self.storage['staleness']) if stale==0]).std()
            rewards_max = torch.cat([rew for rew, stale in zip(self.storage['rewards'], self.storage['staleness']) if stale==0]).max()
            rewards_min = torch.cat([rew for rew, stale in zip(self.storage['rewards'], self.storage['staleness']) if stale==0]).min()

        for i, (rewards, is_truncated, is_terminated, state_vals, terminal_state_vals, advantages, propagated_rewards) in enumerate(zip(
            self.storage['rewards'][::-1],
            self.storage['is_truncateds'][::-1],
            self.storage['is_naturals'][::-1],
            self.storage['state_vals'][::-1],
            self.storage['terminal_state_vals'][::-1],
            self.storage['advantages'][::-1],
            self.storage['propagated_rewards'][::-1])):
            # Reconfigure index
            i = self.get_steps()-i-1
            # Skip if already calculated
            if advantages is None:
                # Reset values and advantages if terminal
                if is_truncated:
                    next_advantages = 0
                    next_state_vals = terminal_state_vals  # 0 for terminal not truncated
                # Compute advantage
                if normalize_rewards: rewards = (rewards - rewards_min) / (rewards_max - rewards_min + 1e-8)
                if moving_standardization is not None: rewards = moving_standardization.apply(rewards.to(moving_standardization.mean.device)).to(self.device)
                deltas = rewards + gamma * next_state_vals - state_vals
                next_advantages = deltas + gamma * gae_lambda * next_advantages
                # Record
                next_state_vals = state_vals
                self.storage['advantages'][i] = next_advantages.copy()
            # Compute rewards
            # Could remove this, but performance impact is minimal
            if propagated_rewards is None:
                if is_truncated:
                    next_rewards = 0
                    if prune is not None: step_num = 0
                next_rewards = rewards + gamma * next_rewards
                self.storage['propagated_rewards'][i] = next_rewards
                # Pruning (only if it's truncated not terminated! store the reason)
                # TODO: Try finite horizon GAE (will still need to prune)
                if prune is not None:
                    self.storage['prunes'][i] = step_num < prune
                    step_num += 1

    # def recompute_state_vals(self, policy):
    #     # NOTE: Could set staleness lower here
    #     with torch.no_grad():
    #         for k1, k2 in zip(('states', 'terminal_states'), ('state_vals', 'terminal_state_vals')):
    #             for i, (keys, states) in enumerate(zip(self.storage['keys'], self.storage[k1])):
    #                 if states is None: continue
    #                 state_split = _utility.processing.split_state(
    #                     self._append_suffix(states, keys=keys), **self.split_kwargs)
    #                 state_split = [t.to(policy.device) for t in state_split]
    #                 self.storage[k2][i] = policy.critic.evaluate_state(state_split).cpu().detach()
    #                 self.storage['advantages'][i] = None
        
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
            if not np.isclose(k_len, k_len[0]).all():
                print(self.storage['keys'])
                print(self.storage['terminal_state_vals'])
            assert np.isclose(k_len, k_len[0]).all(), (
                f'Memory storage out of sync, found storage key lengths {k_len}'
                f' for keys {list(self.storage.keys())}')
        return self.variable_storage['len']
    
    def get_new_len(self, max_per_step=None, invert=False):
        key = ('new_len', max_per_step) if not invert else ('replay_len', max_per_step)
        if key not in self.variable_storage:
            self.variable_storage[key] = 0
            for state, staleness in zip(self.storage['states'], self.storage['staleness']):
                if (staleness==0)^invert:
                    step_len = state.shape[0]
                    if max_per_step is not None: step_len = min(max_per_step, step_len)
                    self.variable_storage[key] += step_len
        return self.variable_storage[key]
    
    def get_replay_len(self, **kwargs):
        return self.get_new_len(invert=True, **kwargs)
    
    def get_new_steps(self, invert=False):
        num_steps = np.sum(np.array(self.storage['staleness'])==0)
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
                k: [v[i] for i in range(len(v)) if self.storage['staleness'][i]==0]
                for k, v in self.storage.items()}
            ret += (storage,)
        else:
            ret += (self.storage,)
            
        if persistent:
            ret += (self.persistent_storage,)

        return ret

    def mark_sampled(self):
        "Mark all steps as sampled"
        for i in range(len(self.storage['staleness'])):
            self.storage['staleness'][i] += 1
        self.clear_var_cache()

    # Cleanup
    def cull_records(self, cull_by_episode=None, max_memories=None):
        "Remove replay memories with reservoir sampling"
        # Parameters
        if cull_by_episode is None: cull_by_episode = self.cull_by_episode
        if max_memories is None: max_memories = self.max_memories
        if max_memories is None: return  # Unlimited memories

        # Cull
        num_memories_to_remove = len(self) - max_memories
        num_memories_removed = 0
        new_steps = np.array(self.storage['staleness']) == 0
        replay_steps = ~new_steps
        steps_to_remove = np.zeros_like(new_steps, dtype=bool)
        while num_memories_to_remove > num_memories_removed:
            # If replay steps remain
            if replay_steps.sum() > 0:
                xx_steps = replay_steps
            # If no replay steps are left
            else:
                warnings.warn(
                    'Insufficient replay memories found to cull, removing new memories instead. Make sure to'
                    ' mark memories after sampling or try raising `max_memories`.')
                xx_steps = new_steps
            idx = np.random.choice(
                np.argwhere(xx_steps).flatten(), 1)[0]
            if cull_by_episode:
                # Find start and end episode idx
                start_idx = idx
                while start_idx > 0 and not self.storage['is_truncateds'][start_idx-1]:
                    start_idx -= 1
                end_idx = idx
                while not self.storage['is_truncateds'][end_idx]:
                    end_idx += 1
                idxs = range(start_idx, end_idx+1)
            else: idxs = [idx]
            for idx in idxs:
                steps_to_remove[idx], xx_steps[idx] = True, False
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

    def cull_stale(self):
        "Cull over-stale memories"
        for i in reversed(range(len(self.storage['staleness']))):
            if self.storage['staleness'][i] > self.max_staleness:
                for k in self.storage:
                    self.storage[k].pop(i)
        self.clear_var_cache()

    def cull_prune(self):
        "Cull memories marked for pruning"
        # Perform pruning
        # NOTE: Doesn't play nicely with whole episode culling
        num_steps = self.get_steps()
        for i, prune in enumerate(self.storage['prunes'][::-1]):
            i = num_steps-i-1
            if prune:
                for k in self.storage:
                    self.storage[k].pop(i)
        self.clear_var_cache()

    def clean_keys(self):
        "Clean all unused keys from suffixes"
        # Get all unique keys
        unique_keys = list(set().union(*self.storage['keys']))
        stored_keys = list(self.persistent_storage['suffixes'])
        for k in stored_keys:
            if k not in unique_keys:
                self.persistent_storage['suffixes'].pop(k)
                self.persistent_storage['target_suffixes'].pop(k)

    def clear(self, clear_persistent=False):
        "Clear all memory"
        for k in self.storage: self.storage[k].clear()
        if clear_persistent:
            for k in self.persistent_storage: self.persistent_storage[k].clear()

    def clear_suffix_cache(self):
        "Clear cache"
        self.persistent_storage['suffix_matrices'].clear()
        self.persistent_storage['target_suffix_matrices'].clear()

    def clear_var_cache(self):
        "Clear computed vars"
        self.variable_storage.clear()

    def cleanup(self):
        "Clean memory"
        self.cull_records()
        self.cull_prune()
        self.cull_stale()
        # self.clean_records()
        self.clean_keys()
        self.clear_suffix_cache()
    
    # Utility
    def _concat_states(self, states, double_batch=False):
        # Pad with duplicate nodes when not sufficient
        # NOTE: Inefficient, nested tensor doesn't have enough
        # functionality yet
        if len(states[0]) == 2:
            # Regular case
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
        # elif (np.array([len(s) for s in states]) == 1).all():
        elif double_batch:
            # Lite case (more memory and compute efficient)
            # NOTE: Shuffle currently incompatible
            # NOTE: Screws with policy indexing, need to use batch with no minibatches 
            states = [(s[0], s[0], np.eye(s[0].shape[0], dtype=bool)) if len(s) == 1 else s for s in states]
            max_self_shape = max(s[0].shape[0] for s in states)
            max_node_shape = max(s[1].shape[0] for s in states)
            s0 = np.zeros((len(states), max_self_shape, states[0][0].shape[-1]))
            for i, s in enumerate(states): s = s[0]; s0[i, :s.shape[0], :s.shape[1]] = s
            s1 = np.zeros((len(states), max_node_shape, states[0][1].shape[-1]))
            for i, s in enumerate(states): s = s[1]; s1[i, :s.shape[0], :s.shape[1]] = s
            # NOTE: The size of `s2` scales at n^2 with the number of cells being simulated, so OOM can be solved by lowering `max_nodes` in dataloader
            s2 = np.ones((len(states), max_self_shape, max_node_shape), dtype=bool)
            for i, s in enumerate(states): s = s[2]; s2[i, :s.shape[0], :s.shape[1]] = s
            states = [s0, s1, s2]
            # states = [
            #     _utility.general.padded_concat(list(map(lambda x: x[0], states)), method='zero'),
            #     _utility.general.padded_concat(list(map(lambda x: x[1], states)), method='zero'),
            #     _utility.general.padded_concat(list(map(lambda x: x[2], states)), method='true')]
            states = [torch.from_numpy(s).to(torch.get_default_dtype()).to(self.device) for s in states]
        else:
            # Lite case (not memory or compute efficient, but closer to previous format for non-lite)
            # Unfold to all 3 representations
            states = [(s[0], s[0], torch.eye(s[0].shape[0], device=self.device)) if len(s) == 1 else s for s in states]
            # Shape, Pad, and concat
            states = [(se.unsqueeze(1), no.expand(se.shape[0], *no.shape), ma.unsqueeze(1)) for se, no, ma in states]
            batch_num = sum(s[0].shape[0] for s in states)
            max_node_shape = max(s[1].shape[1] for s in states)
            max_mask_shape = max(s[2].shape[2] for s in states)
            s0 = torch.concat([s[0] for s in states], dim=0)
            # torch.concat([F.pad(s[1], (0, 0, 0, max_node_shape-s[1].shape[1]), value=0) for s in states], dim=0),  # Wanted to pad with nan, but nan*0=nan
            s1 = torch.zeros((batch_num, max_node_shape, states[0][1].shape[-1]), device=self.device)
            for i, s in enumerate(states):
                s = s[1]
                s1[i*s.shape[0]:(i+1)*s.shape[0], :s.shape[1], :s.shape[2]] = s[0]
            s2 = torch.zeros((batch_num, 1, max_mask_shape), device=self.device)
            # torch.concat([F.pad(s[2], (0, max_mask_shape-s[2].shape[2], 0, 0), value=True) for s in states], dim=0)
            for i, s in enumerate(states):
                s = s[2]
                s2[i*s.shape[0]:(i+1)*s.shape[0], :, :s.shape[2]] = s[0]
            states = [s0, s1, s2]

        return states

    def _append_suffix(
            self,
            state=None,
            *,
            keys,
            suffix_key='suffixes',
            suffix_matrix_key='suffix_matrices'):
        "Append suffixes to state vector with optional cache for common key layouts"
        # Read from cache
        # NOTE: Strings from numpy arrays are slower as keys
        if self.cache_suffix and keys in self.persistent_storage[suffix_matrix_key]:
            suffix_matrix = self.persistent_storage[suffix_matrix_key][keys]
        else:
            # Aggregate suffixes
            suffix_matrix = []
            for k in keys:
                val = self.persistent_storage[suffix_key][k]
                suffix_matrix.append(val)
            suffix_matrix = np.stack(suffix_matrix, axis=0)

            # Add to cache
            if self.cache_suffix:
                self.persistent_storage[suffix_matrix_key][keys] = suffix_matrix
                if len(self.persistent_storage[suffix_matrix_key]) == 101:
                    warnings.warn(
                        f'Persistent storage cache has exceeded 100 entries,'
                        f' please verify that there are over 100 unique environment'
                        f' states in your data ({suffix_matrix_key}).', RuntimeWarning)

        # Append to state
        return np.concat((state, suffix_matrix), axis=1) if state is not None else suffix_matrix

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
