import warnings

import numpy as np
import torch

from . import environment as _environment
from . import policy as _policy
from . import train as _train
from . import utility as _utility


DEFAULT_POLICY_KWARGS = {
    'forward_batch_size': 1_000,
    'vision_size': 1_000}


class BasicManager:
    def __init__(
            self,
            *,
            policy_fname,
            preprocessing_fname,
            mask_fname=None,
            adatas=None,
            spatial=False,
            device='cpu',
            policy_kwargs={},
            preprocessing_kwargs={},
            env_kwargs={}):
        # NOTE: Only plug in input adatas
        # Parameters
        self.device = device
        self.spatial = spatial
        self.policy_kwargs = policy_kwargs
        self.preprocessing_kwargs = preprocessing_kwargs
        self.env_kwargs = env_kwargs

        # Defaults
        temp_kwargs = DEFAULT_POLICY_KWARGS.copy()
        temp_kwargs.update(policy_kwargs)
        policy_kwargs = temp_kwargs

        # Load adatas
        self.adatas = adatas

        # Load policy
        self.policy = _policy.create_agent_from_file(
            policy_fname,
            pinning_spatial=spatial,
            **policy_kwargs).eval().to('cuda')
        self.dim = int(self.policy.positional_dim / 2)

        # Load preprocessing
        self.preprocessing = _utility.processing.Preprocessing(**preprocessing_kwargs).load(preprocessing_fname)

        # Load mask
        self.mask = None
        if mask_fname is not None:
            with _utility.general.open_s3_or_local(mask_fname, 'rb') as f:
                self.mask = np.loadtxt(f).astype(bool)

        # Initialize
        self.env = None
        if adatas is not None: self.set_modalities(adatas, suppress_warning=True)
        self.clear_perturbations()

        # Flags
        self.flags = {
            'ready_to_perturb': False}

        # State storage
        self.states = {}

    # Getters
    def get_mask(self):
        return self.mask

    # Environment management
    def set_modalities(self, adatas, preprocess=True, chunk_size=2_000, suppress_warning=False, strict=False):
        # Create new env if numbers don't match, but preserve state
        # Preprocess if needed
        modalities = [
            _utility.processing.chunk_X(
                ad, chunk_size=chunk_size,
                func=lambda x: self.preprocessing.transform(x, subset_modality=i)[0] if preprocess else None)
            for i, ad in enumerate(adatas)]
        modalities = [torch.tensor(m, device=self.device) for m in modalities]

        # Create new environment
        if self.env is None or len(self.env.modalities) != len(modalities):
            if strict: raise RuntimeError(
                'Unmatched modalities found, cannot replace modalities without initializing new environment')
            if not suppress_warning: warnings.warn('Creating new environment due to mismatched modalities')
            # Preserve old state
            state = self.env.get_state() if self.env is not None else None
            # Initialize new env
            self.env = _environment.EnvironmentBase(
                *modalities, compute_rewards=False, dim=self.dim, **self.env_kwargs).eval().to(self.device)
            # Set old state if applicable
            if state is not None: self.env.set_state(state)
        # Set modalities
        else: self.env.set_modalities(modalities)

        # Update internal variables
        self.adatas = adatas

    def reset_env(self):
        self.env.reset()
        self.flags['ready_to_perturb'] = False

    # State management
    def save_state(self, name):
        self.states[name] = (self.flags['ready_to_perturb'], self.env.get_state())

    def load_state(self, name):
        # Set flag for fresh state, for warnings under perturbation
        self.flags['ready_to_perturb'], state = self.states[name]
        self.env.set_state(state)

    def get_state(self, impute=True, **kwargs):
        # Get state
        pos = self.env.get_positions()
        num_pinning_modules = len(self.policy.pinning)

        # Only impute one
        if isinstance(impute, int) and not isinstance(impute, bool):
            return self.environment_to_features(pos, **kwargs)
        # Impute all
        elif impute:
            return [self.environment_to_features(pos, i, **kwargs) for i in range(num_pinning_modules)]
        # No imputation
        return pos
    
    # Imputation
    def environment_to_features(self, pos, modality='all', *, target_state=None, inverse_preprocess=True, chunk_size=2_000):
        # Parameters
        num_pinning_modules = len(self.policy.pinning)
        num_modalities = len(self.preprocessing.is_sparse_transform)

        # All modalities
        if modality == 'all':
            return [
                self.environment_to_features(
                    pos, modality=i, target_state=target_state,
                    inverse_preprocess=inverse_preprocess, chunk_size=chunk_size)
                for i in range(num_pinning_modules)]

        # Imputes
        with torch.no_grad():
            imputed_pos = self.policy.pinning[modality](pos.to(self.device), Y=target_state).detach().cpu().numpy()
        if inverse_preprocess:
            imputed_pos, = _utility.processing.chunk(
                imputed_pos, chunk_size=chunk_size,
                func=lambda x: self.preprocessing.inverse_transform(x, subset_modality=num_modalities-num_pinning_modules+modality))
        return imputed_pos

    # Simulation
    def simulate(
        self,
        time=512.,
        skip_time=10.,
        store_states='cpu',
        progress_bar=True,
        impute=True,
        impute_kwargs={},
        **kwargs):
        # Parameters
        skip_states = int(skip_time / self.env.delta)

        # Perform simulation
        self.env.set_time(0.).set_max_time(time)
        sim_pos = _train.simulate_until_completion(
            self.env, self.policy, skip_states=skip_states,
            store_states=store_states, progress_bar=progress_bar,
            pbar_total=int(time/self.env.delta), **kwargs)[-1][..., :self.dim]

        # Get sim time
        sim_time = np.arange(0., time, skip_states*self.env.delta)
        if len(sim_time) != sim_pos.shape[0]:
            sim_time = np.concat([sim_time, [self.env.max_time]], axis=0)

        # Flags
        self.flags['ready_to_perturb'] = True
        
        # Impute
        if impute:
            return sim_time, self.environment_to_features(sim_pos, **impute_kwargs)
        return sim_time, sim_pos

    # Perturbations
    def simulate_perturbation(self, time=128., **kwargs):
        # Use perturbations from add_perturbation, but warn if not simulated first without - call `simulate`
        # Warn if not ready
        if not self.flags['ready_to_perturb']:
            # NOTE: Still warns if doing sequential perturbations
            warnings.warn(
                'Environment might not be at steady state, you may be intending to run `.reset_env()` and '
                '`.simulate()` or `.load_state()` before perturbing')
            
        # Simulate
        ret = self.simulate(time=time, **self.hooks, **kwargs)

        # Flags
        self.flags['ready_to_perturb'] = False

        # Return
        return ret

    def add_perturbation(self, features, modality=0, feature_targets=0, clamping='pre', factor=1.):
        # Parameters
        features = np.array(features)
        if not _utility.general.is_list_like(feature_targets):
            feature_targets = [feature_targets for _ in range(len(features))]

        # Get feature idx
        feature_argwheres = [np.argwhere(self.adatas[modality].var_names == feature).flatten() for feature in features]
        feature_occurrences = np.array([argw.shape[0] for argw in feature_argwheres])
        feature_occurrences_under, feature_occurrences_over = feature_occurrences < 1, feature_occurrences > 1
        if feature_occurrences_under.sum() > 0: warnings.warn(f'Features not found, {features[feature_occurrences_under]}')
        if feature_occurrences_over.sum() > 0: warnings.warn(f'Features found multiple times, {features[feature_occurrences_over]}')
        feature_idx = np.concat(feature_argwheres, axis=0)
        feature_targets = sum([argw.shape[0]*[target] for argw, target in zip(feature_argwheres, feature_targets)], [])

        # Get corresponding hook
        if clamping == 'pre':
            self.hooks['env_hooks'].append(
                _utility.hooks.clamp_input_features_hook(
                    feature_idx, self.preprocessing, feature_targets=feature_targets,
                    modality_idx=modality, device=self.device))
        elif clamping == 'post':
            self.hooks['env_hooks'].append(
                _utility.hooks.clamp_inverted_features_hook(
                    feature_idx, self.preprocessing, feature_targets=feature_targets,
                    modality_idx=modality))
        elif clamping == 'action':
            # NOTE: Currently, indexing only works for integration applications (i.e.,
            #       pinning module idx and modality idx are the same)
            self.hooks['env_hooks'].append(
                _utility.hooks.move_toward_targets_hook(
                    feature_idx, self.preprocessing, feature_targets=feature_targets,
                    pinning=self.policy.pinning[modality], modality_idx=modality,
                    factor=factor, device=self.device))

    def clear_perturbations(self):
        self.hooks = {'env_hooks': [], 'action_hooks': []}
