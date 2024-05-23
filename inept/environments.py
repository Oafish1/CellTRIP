import torch

from . import utilities


class trajectory:
    """
    A bounded `dim`-dimensional area where agents control velocity changes
    for individual nodes.

    Rewards
    -------
    'origin': Nodes are rewarded for approaching the origin.
    'target': Nodes are rewarded for approaching a specified target.
    'cosine': Nodes are rewarded for matching inter-node cosine similarities
        across all modalities.
    'euclidean': Nodes are rewarded for matching inter-node euclidean distances
        across all modalities.
    """
    def __init__(self,
        # Data
        *modalities,
        # Params
        dim=2,
        pos_bound=1,
        pos_rand_bound=1,
        vel_bound=1,
        delta=.1,
        # Rewards
        reward_distance=10,
        # reward_origin=1,
        penalty_bound=1,
        penalty_velocity=1,
        penalty_action=1,
        reward_distance_type='euclidean',
        # Device
        device='cpu',
    ):
        # Record modal data
        self.modalities = modalities
        self.dim = dim
        self.pos_bound = pos_bound
        self.pos_rand_bound = pos_rand_bound
        self.vel_bound = vel_bound
        self.delta = delta
        self.reward_distance_type = reward_distance_type
        self.device = device

        # Weights
        self.reward_scales = {
            'reward_distance': reward_distance,     # Emulate distances of each modality
            # 'reward_origin': reward_origin,         # Make sure the mean of positions is close to the origin
            'penalty_bound': penalty_bound,         # Don't touch the bounds
            'penalty_velocity': penalty_velocity,   # Don't move fast
            'penalty_action': penalty_action,       # Don't take drastic action
        }

        # Storage
        self.dist = None

        # Assert all modalities share the first dimension
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]

        # Initialize
        self.reset()

    ### State functions
    def step(self, actions=None, *, delta=None, reward_distance_type=None, return_rewards=False):
        # Defaults
        if actions is None: actions = torch.zeros_like(self.vel)
        if delta is None: delta = self.delta
        if reward_distance_type is None: reward_distance_type = self.reward_distance_type

        # Distance reward
        if reward_distance_type == 'origin': reward_distance = self.get_distance_from_origin()
        elif reward_distance_type == 'target': reward_distance = self.get_distance_from_targets()
        elif reward_distance_type == 'cosine': reward_distance = self.get_distance_match(measure=utilities.cosine_similarity)
        elif reward_distance_type == 'euclidean': reward_distance = self.get_distance_match()

        ## Step positions
        # Add velocity
        self.add_velocities(delta * actions)
        # Iterate positions
        self.pos = self.pos + delta * self.vel
        # Clip by bounds
        self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)
        # Erase velocity of bound-hits
        bound_hit_mask = self.pos.abs() == self.pos_bound
        self.vel[bound_hit_mask] = 0

        # Distance reward
        if reward_distance_type == 'origin': reward_distance -= self.get_distance_from_origin()
        elif reward_distance_type == 'target': reward_distance -= self.get_distance_from_targets()
        elif reward_distance_type == 'cosine': reward_distance -= self.get_distance_match(measure=utilities.cosine_similarity)
        elif reward_distance_type == 'euclidean': reward_distance -= self.get_distance_match()
        reward_distance *= self.reward_scales['reward_distance']

        # Origin reward
        # reward_origin = -self.reward_scales['reward_origin'] * self.pos.mean(dim=0).square().sum()
        # reward_origin = -self.reward_scales['reward_origin'] * self.pos.square().sum(dim=1)

        # Boundary penalty
        penalty_bound = torch.zeros(self.pos.shape[0], device=self.device)
        penalty_bound[bound_hit_mask.sum(dim=1) > 0] = -self.reward_scales['penalty_bound']

        # Velocity penalty
        penalty_velocity = -self.reward_scales['penalty_velocity'] * self.vel.square().mean(dim=1)

        # Action penalty
        penalty_action = -self.reward_scales['penalty_action'] * actions.square().mean(dim=1)

        # Compute total reward
        rwd = reward_distance + penalty_bound + penalty_velocity + penalty_action

        ret = (rwd, self.finished())
        if return_rewards: ret += {
            'distance': reward_distance,
            # 'origin': reward_origin,
            'bound': penalty_bound,
            'velocity': penalty_velocity,
            'action': penalty_action,
        },
        return ret

    def reset(self):
        # Assign random positions and velocities
        self.pos = self.pos_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)
        self.vel = self.vel_bound * 2*(torch.rand((self.num_nodes, self.dim), device=self.device)-.5)

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
        if targets is None: targets = self.modalities[0][:, :2]

        # Calculate distance
        dist = self.pos - targets

        return dist.norm(dim=1)

    def get_distance_match(self, measure=lambda x: utilities.euclidean_distance(x, scaled=True)):
        # Calculate modality distances
        # TODO: Perhaps scale this
        if self.dist is None:
            self.dist = []
            for m in self.modalities:
                m_dist = measure(m)
                self.dist.append(m_dist)

        # Calculate distance for position
        pos_dist = measure(self.pos)

        # Calculate reward
        running = torch.zeros(self.pos.shape[0], device=self.device)
        for dist in self.dist:
            square_ew = (pos_dist - dist)**2
            mean_square_ew = square_ew.mean(dim=1)
            running = running + mean_square_ew
        running = running / len(self.dist)

        return running  # / self.pos_bound**2

    def finished(self):
        return False

    ### Get-Set functions
    def set_modalities(self, modalities):
        self.modalities = modalities

    def set_positions(self, pos):
        self.pos = pos

    def set_velocities(self, vel):
        self.vel = vel

    def get_positions(self):
        return self.pos

    def get_velocities(self):
        return self.vel

    def get_modalities(self):
        return self.modalities

    def get_state(self, include_modalities=False):
        if include_modalities:
            return torch.cat((self.pos, self.vel, *self.modalities), dim=1)
        else:
            return torch.cat((self.pos, self.vel), dim=1)
