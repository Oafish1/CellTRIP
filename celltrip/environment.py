import numpy as np
import torch

from . import utility as _utility


class EnvironmentBase:
    """
    A bounded `dim`-dimensional area where agents control velocity changes
    for individual nodes.
    """
    def __init__(self,
        # Data
        *modalities_or_dataloader,
        # Creation
        dim=16,
        pos_bound=10,
        pos_rand_bound=5,
        vel_bound=1,
        vel_rand_bound=1,
        delta=.1,
        # Targets
        input_modalities=None,  # Which modalities are given as input
        target_modalities=None,  # Which modalities are targets
        # Rewards
        reward_distance=None,
        reward_origin=None,
        penalty_bound=None,
        penalty_velocity=None,
        penalty_action=None,
        # Early stopping
        max_timesteps=1_000,
        min_timesteps=0,
        terminate_time=True,
        terminate_velocity=False,
        vel_threshold=1e-3,
        latency=5,
        # Device
        device='cpu',
        # Extras
        **kwargs,
    ):
        # Record parameters
        self.dim = dim
        self.pos_bound = pos_bound
        self.pos_rand_bound = pos_rand_bound
        self.vel_bound = vel_bound
        self.vel_rand_bound = vel_rand_bound
        self.delta = delta
        self.termination_conds = {
            'time': terminate_time,
            'velocity': terminate_velocity}
        self.vel_threshold = vel_threshold
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        self.latency = latency
        self.input_modalities = input_modalities
        self.target_modalities = target_modalities
        self.device = device

        # Cache
        self.dist = {}

        # Detect if modality input is dataloader
        if (len(modalities_or_dataloader) == 1
            and isinstance(modalities_or_dataloader[0],
                           _utility.processing.PreprocessFromAnnData)):
            self.dataloader, = modalities_or_dataloader
            self.modalities = None
            self.num_modalities = self.dataloader.num_modalities
            self.num_nodes = None
            self.keys = None
        else:
            self.dataloader = None
            self.modalities = modalities_or_dataloader
            self.num_modalities = len(self.modalities)
            assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
            self.num_nodes = self.modalities[0].shape[0]
            self.keys = torch.arange(self.num_nodes)

        # Defaults
        all_modalities = list(range(self.num_modalities))
        if self.target_modalities is None:
            # By default, all modalities will be targets
            self.target_modalities = all_modalities
        elif type(self.target_modalities) == int:
            # Make sure that input is of the correct type
            self.target_modalities = [self.target_modalities]

        if self.input_modalities is None:
            if len(self.target_modalities) == self.num_modalities:
                # By default, all modalities will be inputs if all modalities are targeted
                self.input_modalities = all_modalities
            else:
                # If some modalities aren't targets, they are inputs (imputation)
                self.input_modalities = list(set(all_modalities) - set(self.target_modalities))

        # Weights
        # NOTE: Rewards can and should go positive, penalties can't
        self.reward_scales = {
            'reward_distance': reward_distance,     # Emulate distances of each modality
            'reward_origin': reward_origin,         # Make sure the mean of positions is close to the origin
            'penalty_bound': penalty_bound,         # Don't touch the bounds
            'penalty_velocity': penalty_velocity,   # Don't move fast
            'penalty_action': penalty_action,       # Don't take drastic action
        }

        # Default weights
        reward_not_set = {k: r is None for k, r in self.reward_scales.items()}
        if np.array(list(reward_not_set.values())).all():
            # If all rewards are unset to zero, turn on default final rewards
            self.reward_scales = {
                'reward_distance': 1,
                'reward_origin': 0,
                'penalty_bound': 1,
                'penalty_velocity': 1,
                'penalty_action': 1,
            }
        else:
            # Otherwise, set `None` rewards to zero
            for k in reward_not_set:
                if reward_not_set[k]: self.reward_scales[k] = 0

        # Initialize
        self.reset()
        # self.steps = 0

    def to(self, device):
        self.device = device
        self.pos = self.pos.to(self.device)
        self.vel = self.vel.to(self.device)
        self.modalities = [m.to(self.device) for m in self.modalities]
        return self

    ### State functions
    def step(self, actions=None, *, delta=None, return_itemized_rewards=False):
        # Defaults
        if actions is None: actions = torch.zeros_like(self.vel, device=self.device)
        if delta is None: delta = self.delta

        # Check dimensions
        # assert actions.shape == self.vel.shape

        ### Pre-step calculations
        # Distance reward (Emulate combined intra-modal distances)
        if self.reward_scales['reward_distance'] != 0:
            reward_distance = self.get_distance_match().log()  # CHANGED, ADDED LOG
        else: reward_distance = torch.zeros(actions.shape[0], device=self.device)
        # Origin penalty
        if self.reward_scales['reward_origin'] != 0: reward_origin = self.get_distance_from_origin()
        else: reward_origin = torch.zeros(actions.shape[0], device=self.device)

        ### Step positions
        # Old storage
        # old_vel = self.vel.clone()
        # old_bound_hit_mask = self.pos.abs() == self.pos_bound
        # Add velocity
        self.add_velocities(delta * actions)
        # Iterate positions
        self.pos = self.pos + delta * self.vel
        # Clip by bounds
        self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)
        # Erase velocity of bound-hits
        bound_hit_mask = self.pos.abs() == self.pos_bound
        self.vel[bound_hit_mask] = 0
        # self.pos[bound_hit_mask.sum(dim=1) > 0] = 0  # Send to center
        # Reset cache
        self.reset_cache()

        ### Post-step calculations
        # Finished
        self.timestep += 1
        finished = self.finished()
        # Distance reward
        if self.reward_scales['reward_distance'] != 0:
            reward_distance -= self.get_distance_match().log()  # CHANGED, ADDED LOG
        # Origin reward
        if self.reward_scales['reward_origin'] != 0:
            reward_origin -= self.get_distance_from_origin()
        # Boundary penalty
        # penalty_bound = -(bound_hit_mask*~old_bound_hit_mask).sum(dim=1).float()
        penalty_bound = torch.zeros(self.pos.shape[0], device=self.device)
        penalty_bound[bound_hit_mask.sum(dim=1) > 0] = -1
        # Velocity penalty (Apply to ending velocity)
        penalty_velocity = -self.vel.square().mean(dim=1)  #  * finished
        # penalty_velocity = (old_vel.square() - self.vel.square()).mean(dim=1)
        # Action penalty (Smooth movements)
        penalty_action = -actions.square().mean(dim=1)
        # if finished: print(reason)
        # print(actions)

        ### Management
        if self.get_distance_match().mean() < self.best: self.lapses += delta
        else: self.best = self.get_distance_match().mean(); self.lapses = 0

        # Scale rewards
        # def get_coef_from_step(step, in_step, top_step, out_step=None, factor=5000):
        #     step = step / factor
        #     if step < top_step or out_step is None:
        #         if in_step == top_step: return 1
        #         return np.clip((step - in_step) / (top_step - in_step), 0, 1)
        #     else:
        #         if top_step == out_step: return 1
        #         return np.clip((step - top_step) / (out_step - top_step), 1, 0)

        reward_distance     *=  self.reward_scales['reward_distance']    * 1e2/delta
        reward_origin       *=  self.reward_scales['reward_origin']      * 1e0/delta
        penalty_bound       *=  self.reward_scales['penalty_bound']      * 1e2
        penalty_velocity    *=  self.reward_scales['penalty_velocity']   * 1e0
        penalty_action      *=  self.reward_scales['penalty_action']     * 1e0
        # self.steps += 1

        # Compute total reward
        rwd = (
            reward_distance
            + reward_origin
            + penalty_bound
            + penalty_velocity
            + penalty_action)

        ret = (rwd, finished)
        if return_itemized_rewards: ret += {
            'distance': reward_distance,
            'origin': reward_origin,
            'bound': penalty_bound,
            'velocity': penalty_velocity,
            'action': penalty_action},
        return ret

    def reset(self):
        # Reset modalities if needed
        if self.dataloader is not None:
            modalities, adata_obs, _ = self.dataloader.sample()
            self.set_modalities([
                torch.tensor(m, device=self.device)
                for m in modalities])
            self.keys = adata_obs[0].index.to_numpy()

        # Assign random positions and velocities
        self.pos = self.pos_rand_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)
        self.vel = self.vel_rand_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)

        # Reset timestep
        self.timestep = 0
        self.lapses = 0
        self.best = 0

        # Reset cache
        self.dist.clear()
        self.reset_cache()

    def reset_cache(self):
        self.cached_dist_match = {}

    ### Input functions
    def add_velocities(self, velocities, node_ids=None):
        # Add velocities
        if node_ids is None:
            self.vel = self.vel + velocities
        else:
            self.vel[node_ids] = self.vel[node_ids] + velocities

        # Clip by bounds
        self.vel = torch.clamp(self.vel, -self.vel_bound, self.vel_bound)

    ### Evaluation functions
    def get_distance_match(self, targets=None):
        # Defaults
        if targets is None: targets = self.target_modalities
        targets = tuple(targets)

        # Return if cached
        if targets in self.cached_dist_match: return self.cached_dist_match[targets]

        # Calculate distance for position
        pos_dist = _utility.distance.euclidean_distance(self.pos)

        # Calculate reward
        running = torch.zeros(self.pos.shape[0], device=self.device)
        for target in targets:
            if target not in self.dist: self.calculate_dist([target])
            dist = self.dist[target]
            square_ew = (pos_dist - dist)**2
            mean_square_ew = square_ew.mean(dim=1)
            running = running + mean_square_ew
        running = running / len(targets)

        # Cache result
        self.cached_dist_match[targets] = running

        return running
    
    def get_distance_from_origin(self):
        return self.pos.norm(dim=1)

    def get_distance_from_targets(self, targets=None):
        # Defaults
        if targets is None: targets = self.modalities[0][:, :self.dim]
        
        # Calculate distance
        dist = self.pos - targets

        return dist.norm(dim=1)

    def finished(self):
        # Min threshold for stop
        if self.timestep < self.min_timesteps: return False  # , 'min_thresh', 0.
        # Boundary stop
        # if (self.pos.abs() == self.pos_bound).any(): return True, 'bound', 0.
        # Latency stop
        # if self.lapses >= self.latency: return True, 'lat', 0.
        # Velocity stop
        if self.termination_conds['velocity'] and (self.get_velocities() <= self.vel_threshold).all():
            return True, 'vel', 0.
        # Time stop
        if self.termination_conds['time'] and self.timestep >= self.max_timesteps: return True  # , 'time', 0.
        # Default
        return False  # , 'def', 0.

    ### Get-Set functions
    def calculate_dist(self, targets):
        # Defaults
        if targets is None: targets = self.target_modalities
        targets = tuple(targets)

        for target in targets:
            if target in self.dist: continue
            m = self.modalities[target]
            # NOTE: Only scaled for `self.dist` calculation
            m_dist = _utility.distance.euclidean_distance(m, scaled=True)
            self.dist[target] = m_dist

    def set_modalities(self, modalities):
        # Set modalities and reset pre-calculated inter-node dist
        self.modalities = modalities
        self.calculate_dist(self.target_modalities)

        # Assert all modalities share the first dimension and reset num_nodes
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]

    def set_rewards(self, **new_reward_scales):
        # Check that all rewards are valid
        for k in new_reward_scales:
            if k not in self.reward_scales:
                raise LookupError(f'`{k}` not found in rewards')

        # Set new rewards
        self.reward_scales.update(new_reward_scales)

    def set_termination_conds(self, **new_termination_conds):
        # Check that all rewards are valid
        for k in new_termination_conds:
            if k not in self.termination_conds:
                raise LookupError(f'`{k}` not found in rewards')

        # Set new rewards
        self.termination_conds.update(new_termination_conds)

    def set_positions(self, pos):
        self.pos = pos

    def set_velocities(self, vel):
        self.vel = vel

    def get_positions(self):
        return self.pos

    def get_velocities(self):
        return self.vel
    
    def get_keys(self):
        return self.keys

    def get_target_modalities(self):
        return [m for i, m in enumerate(self.modalities) if i in self.target_modalities]

    def get_return_modalities(self):
        return [m for i, m in enumerate(self.modalities) if i in self.input_modalities]

    def get_modalities(self):
        return self.modalities

    def get_state(self, include_modalities=False, include_timestep=False):
        cat = (self.pos, self.vel)
        if include_timestep: cat += (torch.tensor(self.timestep, device=self.device).expand((self.num_nodes, 1)),)
        if include_modalities: cat += (*self.get_return_modalities(),)
        return torch.cat(cat, dim=-1)
        
    def set_state(self, state):
        # Non-compatible with `include_modalities` or `include_timestep`
        self.set_positions(state[..., :self.dim])
        self.set_velocities(state[..., self.dim:])
