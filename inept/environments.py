import torch


class trajectory:
    def __init__(self, *modalities, dim=2, pos_bound=1, vel_bound=1, delta=.1):
        # Record modal data
        self.modalities = modalities
        self.dim = dim
        self.pos_bound = pos_bound
        self.vel_bound = vel_bound
        self.delta = delta

        # Assert all modalities are the same size
        assert all(m.shape[0] == self.modalities[0].shape[0] for m in self.modalities)
        self.num_nodes = self.modalities[0].shape[0]

        # Assign random positions and velocities
        self.pos = self.pos_bound * 2*(torch.rand((self.num_nodes, self.dim))-.5)
        self.vel = self.vel_bound * 2*(torch.rand((self.num_nodes, self.dim))-.5)

    ### Stepping functions
    def step(self, delta=None):
        # Defaults
        if delta is None: delta = self.delta

        # Iterate positions
        self.pos = self.pos + delta * self.vel

        # Clip by bounds
        self.pos = torch.clamp(self.pos, -self.pos_bound, self.pos_bound)

    ### Input functions
    def add_velocities(self, velocities, node_ids=None):
        # Add velocities
        if node_ids is None:
            self.vel = self.vel + velocities
        else:
            self.vel[node_ids] = self.vel[node_ids] + velocities

        # Clip by bounds
        self.vel = torch.clamp(self.vel, -self.vel_bound, self.vel_bound)

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
