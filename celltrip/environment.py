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
        # Params
        dim=16,
        pos_bound=10,
        pos_rand_bound=1,
        vel_bound=1,
        delta=.1,
        # Early stopping
        vel_threshold=1e-3,
        max_timesteps=1_000,
        # Rewards
        reward_distance=None,
        reward_origin=None,
        penalty_bound=None,
        penalty_velocity=None,
        penalty_action=None,
        # Targets
        modalities_to_return=None,  # Which modalities are given as input
        reward_distance_target=None,  # Which modalities are targets
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
        self.delta = delta
        self.vel_threshold = vel_threshold
        self.max_timesteps = max_timesteps
        self.modalities_to_return = modalities_to_return
        self.reward_distance_target = reward_distance_target
        self.device = device

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
        if self.reward_distance_target is None:
            # By default, all modalities will be targets
            self.reward_distance_target = all_modalities
        elif type(self.reward_distance_target) == int:
            # Make sure that input is of the correct type
            self.reward_distance_target = [self.reward_distance_target]

        if self.modalities_to_return is None:
            if len(self.reward_distance_target) == self.num_modalities:
                # By default, all modalities will be inputs if all modalities are targeted
                self.modalities_to_return = all_modalities
            else:
                # If some modalities aren't targets, they are inputs (imputation)
                self.modalities_to_return = list(set(all_modalities) - set(self.reward_distance_target))

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

        # Storage
        self.dist = None

        # Initialize
        self.reset()

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
        assert actions.shape == self.vel.shape

        ### Pre-step calculations
        # Distance reward
        if self.reward_distance_target == 'debug':
            # Debugging mode to test that PPO works, each cell goes to the position of its first `dim` modal features
            reward_distance = self.get_distance_from_targets()
        else:
            # Emulate combined intra-modal distances
            reward_distance = self.get_distance_match()

        # Origin penalty
        reward_origin = self.get_distance_from_origin()

        ### Step positions
        # Add velocity
        self.add_velocities(delta * actions)
        # Iterate positions
        self.pos = self.pos + delta * self.vel
        # Clip by bounds
        self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)
        # Erase velocity of bound-hits
        bound_hit_mask = self.pos.abs() == self.pos_bound
        self.vel[bound_hit_mask] = 0

        ### Post-step calculations
        # Distance reward
        if self.reward_distance_target == 'debug':
            # Debugging mode to test that PPO works, each cell goes to the position of its first `dim` modal features
            reward_distance -= self.get_distance_from_targets()
        else:
            # Emulate combined intra-modal distances
            reward_distance -= self.get_distance_match()

        # Origin reward
        reward_origin -= self.get_distance_from_origin()
        # reward_origin = -self.reward_scales['reward_origin'] * self.pos.mean(dim=0).square().sum()
        # reward_origin = -self.reward_scales['reward_origin'] * self.pos.square().sum(dim=1)

        # Boundary penalty
        penalty_bound = torch.zeros(self.pos.shape[0], device=self.device)
        penalty_bound[bound_hit_mask.sum(dim=1) > 0] = -1

        # Velocity penalty
        penalty_velocity = -self.vel.square().mean(dim=1)

        # Action penalty
        penalty_action = -actions.square().mean(dim=1)

        ### Management
        # Scale rewards
        reward_distance *=  self.reward_scales['reward_distance']   * 100
        reward_origin *=   self.reward_scales['reward_origin']    * 100
        penalty_bound *=    self.reward_scales['penalty_bound']     * 2
        penalty_velocity *= self.reward_scales['penalty_velocity']  * 10
        penalty_action *=   self.reward_scales['penalty_action']    * 1

        # Compute total reward
        rwd = (
            reward_distance
            + reward_origin
            + penalty_bound
            + penalty_velocity
            + penalty_action
        )

        # Iterate timestep
        self.timestep += 1

        ret = (rwd, self.finished())
        if return_itemized_rewards: ret += {
            'distance': reward_distance,
            'origin': reward_origin,
            'bound': penalty_bound,
            'velocity': penalty_velocity,
            'action': penalty_action,
        },
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
        self.vel = self.vel_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)

        # Reset timestep
        self.timestep = 0

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
    def get_distance_from_origin(self):
        return self.pos.norm(dim=1)

    def get_distance_from_targets(self, targets=None):
        # Defaults
        if targets is None: targets = self.modalities[0][:, :self.dim]
        
        # Calculate distance
        dist = self.pos - targets

        return dist.norm(dim=1)

    def get_distance_match(self, targets=None):
        # Defaults
        if targets is None: targets = self.reward_distance_target
        
        # Calculate modality distances
        # NOTE: Only scaled for `self.dist` calculation
        if self.dist is None:
            self.dist = []
            for target in targets:
                m = self.modalities[target]
                m_dist = _utility.distance.euclidean_distance(m, scaled=True)
                self.dist.append(m_dist)

        # Calculate distance for position
        pos_dist = _utility.distance.euclidean_distance(self.pos)

        # Calculate reward
        running = torch.zeros(self.pos.shape[0], device=self.device)
        for dist in self.dist:
            square_ew = (pos_dist - dist)**2
            mean_square_ew = square_ew.mean(dim=1)
            running = running + mean_square_ew
        running = running / len(self.dist)

        return running

    def finished(self):
        return (self.get_velocities() <= self.vel_threshold).all() or self.timestep >= self.max_timesteps

    ### Get-Set functions
    def set_modalities(self, modalities):
        # Set modalities and reset pre-calculated inter-node dist
        self.modalities = modalities
        self.dist = None

        # Assert all modalities share the first dimension and reset num_nodes
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]

    def set_rewards(self, **new_reward_scales):
        # Check that all rewards are valid
        for k, v in new_reward_scales.items():
            if k not in self.reward_scales:
                raise LookupError(f'`{k}` not found in rewards')

        # Set new rewards
        self.reward_scales.update(new_reward_scales)

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
        return [m for i, m in enumerate(self.modalities) if i in self.reward_distance_target]

    def get_return_modalities(self):
        return [m for i, m in enumerate(self.modalities) if i in self.modalities_to_return]

    def get_modalities(self):
        return self.modalities

    def get_state(self, include_modalities=False):
        if include_modalities:
            return torch.cat((self.pos, self.vel, *self.get_return_modalities()), dim=1)
        else:
            return torch.cat((self.pos, self.vel), dim=1)
        
    def set_state(self, state):
        # Non-compatible with `include_modalities`
        self.set_positions(state[..., :self.dim])
        self.set_velocities(state[..., self.dim:])
