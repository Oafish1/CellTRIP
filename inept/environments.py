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
    def __init__(self, *modalities, dim=2, pos_bound=1, vel_bound=1, delta=.1, reward_type='euclidean', device='cpu'):
        # Record modal data
        self.modalities = modalities
        self.dim = dim
        self.pos_bound = pos_bound
        self.vel_bound = vel_bound
        self.delta = delta
        self.reward_type = reward_type
        self.device = device

        # Storage
        self.dist = None

        # Assert all modalities share the first dimension
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]

        # Initialize
        self.reset()

    ### State functions
    def step(self, actions=None, *, delta=None, reward_type=None):
        # Defaults
        if actions is None: actions = torch.zeros_like(self.vel)
        if delta is None: delta = self.delta
        if reward_type is None: reward_type = self.reward_type

        # Params
        reward_scales = {
            'reward_distance': 10,
            'penalty_bound': 1,
            'penalty_velocity': 1,
            'penalty_action': 1,
        }
        rwd = torch.zeros(self.pos.shape[0])

        # Distance Reward
        if reward_type == 'origin': reward_distance = self.get_distance_from_origin()
        elif reward_type == 'target': reward_distance = self.get_distance_from_targets()
        elif reward_type == 'cosine': reward_distance = self.get_distance_match(measure=utilities.cosine_similarity)
        elif reward_type == 'euclidean': reward_distance = self.get_distance_match(measure=utilities.euclidean_distance)

        # Add velocity
        self.add_velocities(delta * actions)
        # Iterate positions
        self.pos = self.pos + delta * self.vel
        # Clip by bounds
        self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)
        # Erase velocity of bound-hits
        bound_hit_mask = self.pos.abs() == self.pos_bound
        self.vel[bound_hit_mask] = 0

        # Boundary penalty
        penalty_bound = torch.zeros_like(rwd, device=self.device)
        penalty_bound[bound_hit_mask.sum(dim=1) > 0] = reward_scales['penalty_bound']

        # Velocity penalty
        penalty_velocity = torch.zeros_like(rwd, device=self.device)
        penalty_velocity = reward_scales['penalty_velocity'] * self.vel.square().mean(dim=1)

        # Action penalty
        penalty_action = torch.zeros_like(rwd, device=self.device)
        penalty_action = reward_scales['penalty_action'] * actions.square().mean(dim=1)

        # Distance Reward
        if reward_type == 'origin': reward_distance -= self.get_distance_from_origin()
        elif reward_type == 'target': reward_distance -= self.get_distance_from_targets()
        elif reward_type == 'cosine': reward_distance -= self.get_distance_match(measure=utilities.cosine_similarity)
        elif reward_type == 'euclidean': reward_distance -= self.get_distance_match(measure=utilities.euclidean_distance)
        reward_distance *= reward_scales['reward_distance']

        # Compute total reward
        rwd = reward_distance - penalty_bound - penalty_velocity - penalty_action

        return rwd, self.finished()

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

    def get_distance_match(self, measure=utilities.cosine_similarity):
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

        return running

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
