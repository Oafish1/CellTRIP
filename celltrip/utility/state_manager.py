import numpy as np
import torch


def check_requirements(req, kwargs):
    not_found = []
    for s in req:
        if s not in kwargs: not_found.append(s)
    assert len(not_found) == 0, (
        f'All of {not_found} must be passed for `StateManager` call.')
    
    
class StateManager:
    def __init__(self, *, device, **kwargs):
        # NOTE: Assume all input kwargs are options/data
        self.timestep = -1
        self.device = device

    def __call__(self, **kwargs):
        # NOTE: Assume all input kwargs are vars to modify
        # Iterate
        self.timestep += 1
      

class ConvergenceStateManager(StateManager):
    def __init__(
        self,
        *,
        num_nodes=None,
        max_timesteps=1_000,
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Save vars
        self.num_nodes = num_nodes
        self.max_timesteps = max_timesteps

        # Initialize present
        self.present = torch.ones(self.num_nodes, dtype=bool, device=self.device) if num_nodes is not None else None

    def __call__(self, **kwargs):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__call__(**local_vars, **kwargs)

        # Set present
        if self.present is not None: kwargs['present'] = self.present

        # Check requirements
        check_requirements(('present',), kwargs)

        # Modify present
        if self.present is None: kwargs['present'][:] = 1

        return kwargs, self._is_end()

    def _is_end(self):
        return self.timestep+1 >= self.max_timesteps


class DiscoveryStateManager(StateManager):
    def __init__(
        self,
        *,
        discovery,
        num_nodes=None,
        max_timesteps=1_000,
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Save vars
        self.discovery = discovery
        self.num_nodes = num_nodes
        self.max_timesteps = max_timesteps

    def __call__(self, **kwargs):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__call__(**local_vars, **kwargs)

        # Initialize present for timestep 0
        if self.timestep == 0 and 'present' not in kwargs:
            assert self.num_nodes is not None, '`num_nodes` must be defined for automatic `present` generation.'
            kwargs['present'] = torch.zeros(self.num_nodes, dtype=bool, device=self.device)

        # Check requirements
        check_requirements(('present', 'state', 'labels'), kwargs)

        # Copy present to avoid modifying previous
        kwargs['present'] = kwargs['present'].clone()

        # Iterate over each label
        for label, delay, rate, origin in zip(*self.discovery.values()):
            # If delay has been reached
            if self.timestep >= delay:
                # Look at each node
                for i in range(len(kwargs['present'])):
                    # If label matches and not already present
                    if kwargs['labels'][i] == label and not kwargs['present'][i]:
                        # Roll for appearance
                        num_progenitors = ((kwargs['labels']==origin)*kwargs['present'].cpu().numpy()).sum()
                        if np.random.rand() < rate:  # * num_progenitors
                            # Mark as present and set origin if at least one progenitor has spawned
                            if origin is not None and num_progenitors > 0:
                                kwargs['state'][i] = kwargs['state'][np.random.choice(np.argwhere((kwargs['labels']==origin)*kwargs['present'].cpu().numpy()).flatten())]
                            kwargs['present'][i] = True

        # Return
        return kwargs, self._is_end()

    def _is_end(self):
        return self.timestep+1 >= self.max_timesteps


class TemporalStateManager(StateManager):
    def __init__(
        self,
        *,
        temporal,
        num_nodes=None,
        dim=3,
        max_stage_len=500,
        vel_threshold=1e-2,  # 3e-2 for more aggressive culling
        **kwargs,
     ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Save vars
        self.temporal = temporal
        self.num_nodes = num_nodes
        self.dim = dim
        self.max_stage_len = max_stage_len
        self.vel_threshold = vel_threshold

        # Initialize vars
        self.current_stage = -1
        self.stage_start = 0
        self.advance_next = True

    def __call__(self, **kwargs):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__call__(**local_vars, **kwargs)

        # Initialize present for timestep 0
        if self.timestep == 0 and 'present' not in kwargs:
            assert self.num_nodes is not None, '`num_nodes` must be defined for automatic `present` generation.'
            kwargs['present'] = torch.zeros(self.num_nodes, dtype=bool, device=self.device)

        # Check requirements
        check_requirements(('present', 'state', 'times'), kwargs)

        # Update present if needed
        if self.advance_next or self.timestep == 0:
            kwargs['present'] = torch.tensor(np.isin(kwargs['times'], self.temporal['stages'][self.current_stage + 1]))

        # Make change to next stage
        if self.advance_next:
            self.current_stage += 1
            self.stage_start = self.timestep
            self.advance_next = False

        # Initiate change if vel is low
        stage_steps = self.timestep - self.stage_start
        if kwargs['present'].sum() > 0: vel_threshold_met = kwargs['state'][kwargs['present'], self.dim:].square().sum(dim=-1).sqrt().max(dim=-1).values < self.vel_threshold
        else: vel_threshold_met = False

        update = vel_threshold_met or stage_steps >= self.max_stage_len - 1
        if update:
            self.advance_next = True
            if self.current_stage + 1 >= len(self.temporal['stages']): return kwargs, True

        return kwargs, False


class PerturbationStateManager(StateManager):
    def __init__(
        self,
        *,
        perturbation_features=None,
        modal_targets=[],
        num_nodes=None,
        dim=3,
        max_timesteps=1_000,
        max_stage_len=500,
        vel_threshold=None,
        **kwargs,
     ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Save vars
        self.perturbation_features = perturbation_features
        self.modal_targets = modal_targets
        self.num_nodes = num_nodes
        self.dim = dim
        self.max_timesteps = max_timesteps
        self.max_stage_len = max_stage_len
        self.vel_threshold = vel_threshold

        # Initialize vars
        self.current_stage = 0
        self.stage_start = 0
        self.steady_state = None
        self.steady_modalities = None
        self.advance_next = False

    def __call__(self, **kwargs):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'kwargs'): local_vars.pop(k)
        super().__call__(**local_vars, **kwargs)

        # Initialize present for timestep 0
        if self.timestep == 0 and 'present' not in kwargs:
            assert self.num_nodes is not None, '`num_nodes` must be defined for automatic `present` generation.'
            kwargs['present'] = torch.ones(self.num_nodes, dtype=bool, device=self.device)

        # Check requirements
        check_requirements(('state', 'modalities'), kwargs)

        # Advance stage
        if self.advance_next:
            # Record if needed
            if self.current_stage == 0:
                self.steady_state = kwargs['state'].clone()
                self.steady_modalities = tuple([ten.clone() for ten in kwargs['modalities']])

                # Set vel threshold if needed
                if self.vel_threshold is None:
                    # Maybe do this with integration as well?
                    self.vel_threshold = self.steady_state[kwargs['present'], self.dim:].square().sum(dim=-1).sqrt().max(dim=-1).values

            # Make meta changes
            self.current_stage += 1
            self.stage_start = self.timestep
            self.advance_next = False

            # Set to steady state
            kwargs['state'] = self.steady_state.clone()
            kwargs['modalities'] = tuple([ten.clone() for ten in self.steady_modalities])

            # Modify feature
            target_modality, target_feature = self.perturbation_feature_pairs[self.current_stage - 1]
            # kwargs['modalities'][self.modal_inputs[target_modality]][:, target_feature] = kwargs['modalities'][target_modality][:, target_feature].mean()  # Revert to mean
            kwargs['modalities'][self.modal_inputs[target_modality]][:, target_feature] = 0  # Knockdown

        # Initial setup
        if self.timestep == 0:
            # Base case for modal inputs
            self.modal_inputs = np.array([i for i in range(len(kwargs['modalities'])) if i not in self.modal_targets])

            # Case for integration
            if len(self.modal_inputs) == 0: self.modal_inputs = list(range(len(kwargs['modalities'])))
            
            # Calculate feature idx on first run
            if self.perturbation_features is None: self.perturbation_features = [np.arange(m.shape[1]) for i, m in enumerate(kwargs['modalities']) if i in self.modal_inputs]
            self.perturbation_feature_pairs = [(i, f) for i, fs in enumerate(self.perturbation_features) for f in fs]
            self.num_features = sum([len(fs) for fs in self.perturbation_features])

        # Calculate advance criterion
        if self.vel_threshold is not None:
            vel_threshold_met = kwargs['state'][kwargs['present'], self.dim:].square().sum(dim=-1).sqrt().max(dim=-1).values < self.vel_threshold
        else: vel_threshold_met = False
        max_len = self.max_timesteps if self.current_stage == 0 else self.max_stage_len
        time_threshold_met = self.timestep - self.stage_start >= max_len - 1
        # update = (vel_threshold_met and self.current_stage != 0) or time_threshold_met
        update = vel_threshold_met or time_threshold_met

        if update:
            self.advance_next = True
            if self.current_stage + 1 > self.num_features: return kwargs, True
        
        return kwargs, False