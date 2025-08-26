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
        dim=8,
        pos_bound=torch.inf,
        pos_rand_bound=1,
        vel_bound=1,
        vel_rand_bound=1,
        uniform_bounds=False,
        force_bound=torch.inf,
        discrete_force=1,
        friction_force=0,
        delta=.1,  # .2
        discrete=False,
        spherical=False,
        # Targets
        input_modalities=None,  # Which modalities are given as input
        target_modalities=None,  # Which modalities are targets
        noise_std=.1,  # Noise to apply to input and/or target modalities
        input_noise=False,
        target_noise=False,
        # Rewards
        compute_rewards=True,
        reward_distance=None,
        reward_pinning=None,
        lin_deg=1,  # What degree to compute for least squares (Experimental)
        reward_origin=None,
        penalty_bound=None,
        penalty_velocity=None,
        penalty_action=None,
        epsilon=1e-3,
        # Early stopping
        min_time=2**6,
        max_time=2**7,
        eval_time=2**5,  # [2**5, 2**6]
        terminate_min_time=True,
        terminate_max_time=True,
        terminate_random=True,
        terminate_bound=False,
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
        self.uniform_bounds = uniform_bounds
        self.force_bound = force_bound
        self.discrete_force = discrete_force
        self.friction_force = friction_force
        self.delta = delta
        self.discrete = discrete
        self.spherical = spherical
        self.termination_conds = {
            'min_time': terminate_min_time,
            'max_time': terminate_max_time,
            'random': terminate_random,
            'velocity': terminate_velocity,
            'bound': terminate_bound}
        self.eval_time = eval_time
        self.vel_threshold = vel_threshold
        self.epsilon = epsilon
        self.noise_std = noise_std
        self.input_noise = input_noise
        self.target_noise = target_noise
        self.compute_rewards = compute_rewards
        self.max_time = max_time
        self.min_time = min_time
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
        self.lin_deg = lin_deg
        self.reward_scales = {
            'reward_distance': reward_distance,     # Emulate distances of each modality
            'reward_pinning': reward_pinning,     # Explained variance
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
                'reward_distance': 0,
                'reward_pinning': 1,
                'reward_origin': 0,
                'penalty_bound': 0,
                'penalty_velocity': 1,
                'penalty_action': 1,
            }
        else:
            # Otherwise, set `None` rewards to zero
            for k in reward_not_set:
                if reward_not_set[k]: self.reward_scales[k] = 0

        # Initialize
        self.reset(resample=True, renoise=True)
        self.stored_changes = {}
        # self.steps = 0

    def to(self, device):
        self.device = device
        self.pos = self.pos.to(self.device)
        self.vel = self.vel.to(self.device)
        self.modalities = [m.to(self.device) for m in self.modalities]
        self.noise = [m.to(self.device) if isinstance(m, torch.Tensor) else m for m in self.noise]
        return self
    
    def train(self, clear=True):
        # Restore all conditions and clear stored memory
        self.restore_vars(clear=clear)

        return self
    
    def eval(self, **kwargs):
        # Store vars
        self.store_vars(**kwargs)
        # Remove random termination conditions
        self.termination_conds['random'] = False
        # Make max length 5x
        self.max_time *= 5
        # Set noise to zero
        self.set_noise_std(0)
        self.noise = len(self.modalities)*[0]

        return self
    
    def restore_vars(self, clear=False):
        # Return if empty
        if len(self.stored_changes) == 0: return self

        # Restore from memory
        self.termination_conds = self.stored_changes['termination_conds']
        self.max_time = self.stored_changes['max_time']
        self.set_noise_std(self.stored_changes['noise_std'])
        if 'noise' in self.stored_changes:
            self.noise = self.stored_changes['noise']
        if clear: self.stored_changes.clear()

        return self
    
    def store_vars(self, store_noise=False):
        self.stored_changes['termination_conds'] = self.termination_conds
        self.stored_changes['max_time'] = self.max_time
        self.stored_changes['noise_std'] = self.noise_std
        if store_noise: self.stored_changes['noise'] = self.noise

        return self


    ### State functions
    def step(self, actions=None, *, delta=None, pinning_func_list=None, return_itemized_rewards=False):
        # Defaults
        if actions is None: actions = torch.zeros_like(self.vel, device=self.device)
        if delta is None: delta = self.delta

        # Check dimensions
        # assert actions.shape == self.vel.shape

        # Hypersphere or hypercube constraints
        # NOTE: Hypersphere is technically more correct, but also harder on the model

        ### Pre-step calculations
        if self.compute_rewards:
            # Distance reward (Emulate combined intra-modal distances)
            get_reward_distance = lambda: self.get_distance_match(use_cache=True)
            # get_reward_distance = lambda: (self.get_distance_match()+self.epsilon).log().mean(dim=-1)
            # get_reward_distance = lambda: 1 / (1+self.get_distance_match())
            if self.reward_scales['reward_distance'] != 0:
                reward_distance = get_reward_distance()
                # reward_distance = 0
            else: reward_distance = torch.zeros(actions.shape[0], device=self.device)
            # Pinning reward
            get_reward_pinning = lambda: self.get_pinning(use_cache=True, pinning_func_list=pinning_func_list)
            # get_reward_pinning = lambda: (self.get_pinning(use_cache=True)+self.epsilon).log().mean(dim=-1)
            if self.reward_scales['reward_pinning'] != 0:
                reward_pinning = get_reward_pinning()
                # reward_pinning = 0
            else: reward_pinning = torch.zeros(actions.shape[0], device=self.device)
            # Origin penalty
            get_reward_origin = lambda: self.get_distance_from_origin()
            # get_reward_origin = lambda: (self.get_distance_from_origin()+self.epsilon).log()
            if self.reward_scales['reward_origin'] != 0:
                reward_origin = get_reward_origin()
                # reward_origin = 0
            else: reward_origin = torch.zeros(actions.shape[0], device=self.device)
            # Velocity penalty
            if self.spherical:
                get_penalty_velocity = lambda: self.vel.norm(dim=-1)
                # get_penalty_velocity = lambda: (self.vel.norm(dim=-1)+self.epsilon).log()
            else:
                # get_penalty_velocity = lambda: self.vel.square().mean(dim=-1)
                get_penalty_velocity = lambda: self.vel.abs().mean(dim=-1)
                # get_penalty_velocity = lambda: (self.vel.square().mean(dim=-1)+self.epsilon).log()
                # get_penalty_velocity = lambda: self.vel.mean(dim=-1)
            if self.reward_scales['penalty_velocity'] != 0:
                penalty_velocity = get_penalty_velocity()
                # penalty_velocity = 0
            else: penalty_velocity = torch.zeros(actions.shape[0], device=self.device)

        ### Step positions
        # Old storage
        # old_vel = self.vel.clone()
        # old_bound_hit_mask = self.pos.abs() == self.pos_bound
        # Clamp actions
        # actions = actions.clamp(-self.force_bound, self.force_bound)
        # Convert actions to velocity
        # NOTE: It would be nice to use polar here, but couldn't find a general method which didn't require unreasonable precision for later axes
        # magnitude = (.5*actions[..., [-1]]+.5).clamp(0, self.force_bound)
        # direction = actions[..., :-1] / actions[..., :-1].norm(keepdim=True, dim=-1)
        # force = magnitude * direction
        if self.discrete:
            force = self.discrete_force * (actions - 1)
            if self.spherical:
                force_norm = force.norm(keepdim=True, dim=-1)
                force_norm[force_norm==0] = 1
                force = self.discrete_force * force / force_norm
        else: force = actions
        if self.spherical:
            force_norm = force.norm(keepdim=True, dim=-1)
            strong_mask = force_norm.squeeze(-1) > self.force_bound
            force[strong_mask] = self.force_bound * force[strong_mask] / force_norm[strong_mask]
        else:
            force = force.clamp(-self.force_bound, self.force_bound)
        # Add velocity and apply friction
        self.add_velocities(delta * force)
        self.apply_friction(realized_force=delta*self.friction_force)
        # Iterate positions
        self.pos = self.pos + delta * self.vel  # .square()  # TODO: Experimental square
        # Clip by bounds
        # self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)
        # Adjust nodes on bound-hits
        # bound_hit_mask = self.pos.abs() == self.pos_bound
        # self.pos[bound_hit_mask.sum(dim=1) > 0] = 0  # Send to center
        # self.vel[bound_hit_mask] = 0  # Kill velocity
        # self.vel[bound_hit_mask] = -self.vel[bound_hit_mask]  # Bounce
        # Out of bounds
        if self.spherical: oob = self.pos.norm(dim=-1)-self.pos_bound
        else: oob = (self.pos.abs() >= self.pos_bound).sum(dim=-1)
        # oob[oob < 0] = 0
        self.vel[oob > 0] = 0
        if self.spherical:
            self.pos[oob > 0] = self.pos_bound * self.pos[oob > 0] / self.pos[oob > 0].norm(keepdim=True, dim=-1)
        else: self.pos = self.pos.clamp(-self.pos_bound, self.pos_bound)
        # reward_distance[oob > 0] = 0
        # Reset cache
        self.reset_cache()

        ### Post-step calculations
        # Finished
        self.time += delta
        finished, _ = self.finished()
        if self.compute_rewards:
            # Distance reward
            if self.reward_scales['reward_distance'] != 0: reward_distance -= get_reward_distance()
            # Pinning reward
            if self.reward_scales['reward_pinning'] != 0: reward_pinning -= get_reward_pinning()
            # Origin reward
            if self.reward_scales['reward_origin'] != 0: reward_origin -= get_reward_origin()
            # Velocity penalty (Apply to ending velocity)
            if self.reward_scales['penalty_velocity'] != 0: penalty_velocity -= get_penalty_velocity()
            # Boundary penalty
            penalty_bound = torch.zeros(self.pos.shape[0], device=self.device)
            penalty_bound[oob > 0] = -1
            # Action penalty (Smooth movements)
            # NOTE: Calculated on unclipped forces
            if self.spherical:
                # penalty_action = -force_norm.squeeze(-1).square()  # WRONG
                penalty_action = -force.norm(dim=-1)
            else: penalty_action = -actions.square().mean(dim=-1)

            ### Management
            if self.get_distance_match().mean() < self.best: self.lapses += delta
            else: self.best = self.get_distance_match().mean(); self.lapses = 0

            reward_distance     *=  self.reward_scales['reward_distance']    * 1e-1/delta
            reward_pinning      *=  self.reward_scales['reward_pinning']     * 5e-1/delta  # 1e-6
            reward_origin       *=  self.reward_scales['reward_origin']      * 1e-1/delta
            penalty_bound       *=  self.reward_scales['penalty_bound']      * 1e0
            penalty_velocity    *=  self.reward_scales['penalty_velocity']   * 1e-1/delta  # 1e-3
            penalty_action      *=  self.reward_scales['penalty_action']     * 1e-3
            # self.steps += 1
        else:
            placeholder = torch.zeros(self.num_nodes, device=self.device)
            reward_distance = placeholder
            reward_pinning = placeholder
            reward_origin = placeholder
            penalty_bound = placeholder
            penalty_velocity = placeholder
            penalty_action = placeholder

        # Compute total reward
        rwd = (
            reward_distance
            + reward_pinning
            + reward_origin
            + penalty_bound
            + penalty_velocity
            + penalty_action)
        
        # Return
        steady = np.abs(self.time - self.eval_time) < self.epsilon
        # steady = self.time >= self.eval_time
        # steady = (self.time >= self.eval_time[0]) and (self.time <= self.eval_time[1])
        ret = (rwd, steady, finished)
        if return_itemized_rewards: ret += {
            'distance': reward_distance,
            'pinning': reward_pinning,
            'origin': reward_origin,
            'bound': penalty_bound,
            'velocity': penalty_velocity,
            'action': penalty_action},
        return ret

    def reset(self, resample=True, renoise=True, **kwargs):
        # Reset modalities if needed
        if resample and self.dataloader is not None:
            modalities, adata_obs, _ = self.dataloader.sample(**kwargs)
            self.set_modalities([
                torch.tensor(m, device=self.device)
                for m in modalities],
                keys=adata_obs[0].index.to_numpy())
            # Add randomness to key to avoid culling in memory
            if self.noise_std > 0 and (self.input_noise or self.target_noise):
                noise_id = np.random.randint(2**32)
                self.keys = [(k, noise_id) for k in self.keys]

        # Generate noise
        if renoise: self.noise = [self.noise_std*torch.randn_like(m, device=self.device) for m in self.modalities]

        # Assign random positions and velocities
        shape = (self.num_nodes, self.dim)
        if self.uniform_bounds:
            direction = 2*torch.rand(shape, device=self.device)-1
            if self.spherical: direction /= direction.norm(keepdim=True, dim=-1)
            magnitude = self.pos_rand_bound * torch.rand((self.num_nodes, 1), device=self.device)
            self.pos = magnitude * direction
            direction = 2*torch.rand(shape, device=self.device)-1
            if self.spherical: direction /= direction.norm(keepdim=True, dim=-1)
            magnitude = self.vel_rand_bound * torch.rand((self.num_nodes, 1), device=self.device)
            self.vel = magnitude * direction
        else:
            self.pos = self.pos_rand_bound*torch.randn(shape, device=self.device)
            self.vel = self.vel_rand_bound*torch.randn(shape, device=self.device)

        # Determine randomizer
        randomizer = (lambda *args, **kwargs: 2*torch.rand(*args, **kwargs)-1) if self.uniform_bounds else torch.randn

        # Perform randomization
        if not self.spherical:
            self.pos = self.pos_rand_bound * randomizer(shape, device=self.device)
            self.vel = self.vel_rand_bound * randomizer(shape, device=self.device)
        else:
            direction = torch.randn(shape, device=self.device)
            direction /= direction.norm(keepdim=True, dim=-1)
            magnitude = self.pos_rand_bound * randomizer((self.num_nodes, 1), device=self.device)
            self.pos = magnitude * direction
            direction = torch.randn(shape, device=self.device)
            direction /= direction.norm(keepdim=True, dim=-1)
            magnitude = self.vel_rand_bound * randomizer((self.num_nodes, 1), device=self.device)
            self.vel = magnitude * direction

        # self.pos = self.pos_rand_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)
        # self.vel = self.vel_rand_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)

        # Reset time
        self.time = 0
        self.lapses = 0
        self.best = 0

        # Reset end cond
        self.end_time = np.random.randint(self.min_time, self.max_time) if self.min_time < self.max_time else self.max_time

        # Reset cache
        self.dist.clear()
        self.reset_cache()

        return self

    def reset_cache(self):
        self.cached_dist_match = {}
        self.cached_pinning = {}

    ### Input functions
    def add_velocities(self, velocities, node_ids=None):
        # Add velocities
        if node_ids is None:
            self.vel = self.vel + velocities
        else:
            self.vel[node_ids] = self.vel[node_ids] + velocities

        # Clamp
        if self.spherical:
            vel_norm = self.vel.norm(dim=-1)
            fast_mask = vel_norm > self.vel_bound
            self.vel[fast_mask] = self.vel_bound * self.vel[fast_mask] / vel_norm[fast_mask].unsqueeze(-1)
        else:
            self.vel = torch.clamp(self.vel, -self.vel_bound, self.vel_bound)

    def apply_friction(self, realized_force=None):
        # Defaults
        if realized_force is None: realized_force = self.delta * self.friction_force

        # Apply friction
        if self.spherical:
            vel_norms = self.vel.norm(keepdim=True, dim=-1) + self.epsilon
            vel_units = self.vel / vel_norms
            new_vel_norms = torch.clip(vel_norms-realized_force, min=0)
            self.vel = new_vel_norms * vel_units
        else:
            mag, sig = self.vel.abs(), self.vel.sign()
            self.vel = sig * torch.clip(mag-realized_force, min=0)
        

    ### Evaluation functions
    def get_distance_match(self, targets=None, use_cache=False, log='per-modality', mean=True):
        """
        Get distance error between latent space and actual data.
        log:
            `per-modality` - Log for each modality
            `pre-mean` - Log before mean but after aggregated modalities
            `post-mean` - Log of mean
        """
        # Defaults
        if targets is None: targets = self.target_modalities
        targets = tuple(targets)

        # Return if cached
        if use_cache and targets in self.cached_dist_match: return self.cached_dist_match[targets]

        # Calculate distance for position
        pos_dist = _utility.distance.euclidean_distance(self.pos)  # , scaled=not self.spherical

        # Calculate reward
        running = torch.zeros(self.pos.shape[0], device=self.device)
        for target in targets:
            if target not in self.dist: self.calculate_dist([target])
            dist = self.dist[target]
            square_ew = (pos_dist - dist)**2
            if log == 'per-modality':
                square_ew = (square_ew+self.epsilon).log()
            running = running + square_ew
        running = running / len(targets)
        if log == 'pre-mean':
            running = (running+self.epsilon).log()
        running.fill_diagonal_(0)
        if mean: running = running.mean(dim=-1)
        if log == 'post-mean': running = (running+self.epsilon).log()

        # Cache result
        self.cached_dist_match[targets] = running

        return running
    
    def get_pinning(self, targets=None, pinning_func_list=None, use_cache=False):
        # Defaults
        if targets is None: targets = self.target_modalities
        targets = tuple(targets)

        # Return if cached
        if use_cache and targets in self.cached_pinning: return self.cached_pinning[targets]

        # Calculate least squares classification error
        running_mse = 0
        for i, target in enumerate(targets):
            m = self.modalities[target]
            if self.target_noise: m = m + self.noise[target]
            if pinning_func_list is None:
                A, B = torch.concat([self.pos.pow(deg+1) for deg in range(self.lin_deg)] + [torch.ones((self.pos.shape[0], 1), device=self.device)], dim=-1), m  # Could speed up a little by keeping ones vec, but not too expensive
                # Match A norm to B norm
                # A_norm = torch.linalg.matrix_norm(self.pos) * np.sqrt(B.shape[1]/self.pos.shape[1])
                # B_norm = torch.linalg.matrix_norm(B)
                # Make mean A var 1 over all dims (Doesn't converge)
                # A_norm = self.pos.var(dim=0).mean(dim=-1)
                # B_norm = 1
                # No adjustment
                A_norm = B_norm = 1
                # Perform lstsq
                X = torch.linalg.lstsq(A, B / B_norm).solution
                err = torch.matmul(A, X) * A_norm - B
            else:
                err = m - pinning_func_list[i](self.pos, m, input_standardization=True, output_standardization=False).detach()
                # err = pinning_func_list[i].output_standardization.apply(m) - pinning_func_list[i](self.pos, input_standardization=True).detach()
            mse = err.square().mean(dim=-1)
            # mse = (err.square() + self.epsilon).log().mean(dim=-1)
            # mse = (err.square() + self.epsilon).mean(dim=-1).log()
            # mse /= m.square().mean()  # Scale for fairness
            mse = -1 / (1 + mse)  # Transform
            running_mse += mse / len(targets)

        # Cache result
        self.cached_pinning[targets] = running_mse
        # print(running_mse)

        return running_mse
    
    def get_distance_from_origin(self):
        if self.spherical: return self.pos.norm(dim=-1)
        else: return self.pos.square().mean(dim=-1)

    def get_distance_from_targets(self, targets=None):
        # Defaults
        if targets is None: targets = self.modalities[0][:, :self.dim]
        
        # Calculate distance
        dist = self.pos - targets

        return dist.norm(dim=1)

    def finished(self):
        # Min threshold for stop
        if self.termination_conds['min_time'] and self.time < min(self.min_time, self.max_time): return False, 'min_time'
        # Boundary stop
        if self.termination_conds['bound'] and (self.pos.norm(dim=-1)-self.pos_bound > 0).any(): return True, 'bound'
        # Latency stop
        # if self.lapses >= self.latency: return True, 'lat'
        # Velocity stop
        # velocity_cond = self.get_velocities().norm(dim=-1) if self.spherical else self.get_velocities().square().mean(dim=-1)
        # if self.termination_conds['velocity'] and (velocity_cond <= self.vel_threshold).all(): return True, 'vel'
        velocity_cond = self.get_velocities().norm(dim=-1).mean() if self.spherical else self.get_velocities().square().mean(dim=-1).mean()
        if self.termination_conds['velocity'] and velocity_cond <= self.vel_threshold: return True, 'vel'
        # Time stop
        if self.termination_conds['max_time'] and self.time >= self.max_time: return True, 'max_time'
        # Random stop
        if self.termination_conds['random'] and self.time >= self.end_time: return True, 'rand'
        # Default
        return False, 'def'

    ### Get-Set functions
    def calculate_dist(self, targets):
        # Defaults
        if targets is None: targets = self.target_modalities
        targets = tuple(targets)

        for target in targets:
            if target in self.dist: continue
            m = self.modalities[target]
            if self.target_noise: m += self.noise[target]
            # NOTE: Only scaled for `self.dist` calculation
            m_dist = _utility.distance.euclidean_distance(m, scaled=True)
            self.dist[target] = m_dist

    def disable_rewards(self):
        self.compute_rewards = False
        return self

    def enable_rewards(self):
        self.compute_rewards = True
        return self

    def set_delta(self, delta):
        self.delta = delta
        return self

    def set_noise_std(self, noise_std):
        self.noise_std = noise_std
        return self

    def set_modalities(self, modalities, keys=None):
        # Set modalities
        self.modalities = modalities

        # Set keys
        self.keys = keys if keys is not None else list(range(modalities[0].shape[0]))
        assert len(self.keys) == modalities[0].shape[0], 'Modal length must match key array length'

        # Reset pre-calculated inter-node dist
        # self.calculate_dist(self.target_modalities)

        # Assert all modalities share the first dimension and reset num_nodes
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]
        
        return self

    def set_rewards(self, **new_reward_scales):
        # Check that all rewards are valid
        for k in new_reward_scales:
            if k not in self.reward_scales:
                raise LookupError(f'`{k}` not found in rewards')

        # Set new rewards
        self.reward_scales.update(new_reward_scales)

        return self

    def set_termination_conds(self, exclusive=False, **new_termination_conds):
        # Check that all rewards are valid
        for k in new_termination_conds:
            if k not in self.termination_conds:
                raise LookupError(f'`{k}` not found in rewards')
            
        # Exclusive case
        if exclusive:
            for k in self.termination_conds:
                if k not in new_termination_conds:
                    new_termination_conds[k] = False

        # Set new rewards
        self.termination_conds.update(new_termination_conds)

        return self

    def set_positions(self, pos):
        self.pos = pos

    def set_velocities(self, vel):
        self.vel = vel

    def get_positions(self):
        return self.pos

    def get_velocities(self):
        return self.vel
    
    def get_keys(self, noise=True, stringify=True):
        if not noise and (self.noise_std > 0 and (self.input_noise or self.target_noise)):
            return list(map(lambda x: x[0], self.keys))
        if stringify: return list(map(str, self.keys))
        else: return self.keys

    def get_target_modalities(self, **kwargs):
        return self.get_modalities(**kwargs, _indices=self.target_modalities)

    def get_input_modalities(self, **kwargs):
        return self.get_modalities(**kwargs, _indices=self.input_modalities)

    def get_modalities(self, noise=True, _indices=None):
        if _indices is None: _indices = list(range(len(self.modalities)))
        return [m+n if noise else m for i, (m, n) in enumerate(zip(self.modalities, self.noise)) if i in _indices]

    def get_state(self, include_modalities=False, include_time=False):
        cat = (self.pos, self.vel)  # (self.pos/self.pos_bound, self.vel/self.vel_bound)
        if include_time: cat += (torch.tensor(self.time, device=self.device).expand((self.num_nodes, 1)),)
        if include_modalities: cat += (*self.get_input_modalities(noise=self.input_noise),)
        return torch.cat(cat, dim=-1)
        
    def set_state(self, state):
        # Non-compatible with `include_modalities` or `include_time`
        self.set_positions(state[..., :self.dim])
        self.set_velocities(state[..., self.dim:])
