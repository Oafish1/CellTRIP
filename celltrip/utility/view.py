import matplotlib
from matplotlib.ticker import ScalarFormatter
import mpl_toolkits.mplot3d as mp3d
import numpy as np
import sklearn.decomposition
import torch

from .. import utility as _utility


class ViewBase:
    def __init__(
        self,
        # Data
        # None
        *args,
        # Data params
        # None
        # Arguments
        ax,
        # Styling
        # None
        **kwargs,
    ):
        # Arguments
        self.ax = ax

        # Initialize plots
        pass

    def update(self, frame):
        # Update plots per frame
        pass


class ViewModalDistBase(ViewBase):
    "Class which controls a plot showing a live 3D view of the environment"
    def __init__(
        self,
        # Data
        states,
        modalities,
        *args,
        # Data params
        modal_targets,
        partitions=None,
        # Arguments
        # None
        # Styling
        # None
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Storage
        self.states = states
        self.modalities = modalities

        # Calculate modal dist
        # NOTE: Only calculated for targets
        self.modal_dist = []
        for target in modal_targets:
            m = self.modalities[target].cpu()
            m_dist = _utility.distance.partition_distance(m, partitions=partitions, func=lambda x: _utility.distance.euclidean_distance(x, scaled=True))
            self.modal_dist.append(m_dist)


class ViewLinesBase(ViewModalDistBase):
    "Class which controls a plot showing a live 3D view of the environment"
    def __init__(
        self,
        # Data
        states,
        modalities,
        *args,
        # Data params
        dim,
        modal_targets,
        # Arguments
        seed=None,
        # Styling
        num_lines=300,  # Number of attraction and repulsion lines
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Storage
        self.states = states
        self.modalities = modalities

        # Data params
        self.dim = dim

        # Preserve random state
        if seed is not None:
            prev_rand_state = np.random.get_state()
            np.random.seed(seed)

        # Get upper triangular random indices efficiently - Select lines to show
        self.line_indices = []
        for _ in range(len(self.modal_dist)):
            self.line_indices.append(np.random.randint(self.modalities[0].shape[0], size=2*num_lines).reshape((-1, 2)))
            for i in range(self.line_indices[-1].shape[0]):
                # Iterate until unique and upper triangular
                while True:
                    # Upper triangular for easier duplicate detection
                    if self.line_indices[-1][i][0] > self.line_indices[-1][i][1]:
                        self.line_indices[-1][i] = self.line_indices[-1][i][::-1]
                    # Reroll if equal or duplicated
                    elif self.line_indices[-1][i][0] == self.line_indices[-1][i][1] or (self.line_indices[-1] == self.line_indices[-1][i]).all(axis=-1).sum() > 1:
                        self.line_indices[-1][i] = np.random.randint(self.modalities[0].shape[0], size=2)
                    # Escape if all requirements met
                    else: break

        # Reset random state
        if seed is not None: np.random.set_state(prev_rand_state)

        # Style
        def get_rgba_from_dd_array(dd_array, visible=None, min_alpha=0, max_value_alpha=2):
            # Handle sparse case
            if isinstance(max_value_alpha, np.matrix): max_value_alpha = torch.Tensor(max_value_alpha).squeeze()

            # Determine colors
            color = np.array([(0., 0., 1.) if dd > 0 else (1., 0., 0.) for dd in dd_array])
            alpha = np.expand_dims(np.clip(np.abs(dd_array), min_alpha * max_value_alpha, max_value_alpha) / max_value_alpha, -1)
            if visible is not None: alpha[~np.array(visible)] = 0.
            return np.concatenate((color, alpha), axis=-1)
        self.get_rgba_from_dd_array = get_rgba_from_dd_array


class View3D(ViewLinesBase):
    "Class which controls a plot showing a live 3D view of the environment"
    def __init__(
        self,
        # Data
        present,
        states_3d,
        rewards,
        modalities,
        labels,
        *args,
        # Data params
        dim,
        partitions=None,
        # Arguments
        ax,
        interval,  # Time between frames
        skip=1,
        # Styling
        ms=6,
        lw=2,
        rotations_per_second=.1,  # Camera azimuthal rotations per second
        arrow_length_scale=1,
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Storage
        self.present = present
        self.states_3d = states_3d if states_3d is not None else self.states
        self.dim_3d = dim if states_3d is None else 3
        self.rewards = rewards
        self.modalities = modalities
        self.labels = labels

        # Data params
        self.partitions = partitions

        # Arguments
        self.skip = skip

        # Styling
        self.arrow_length_scale = arrow_length_scale

        # Initialize nodes
        self.nodes = [
            self.ax.plot(
                [], [],
                linestyle='',
                markeredgecolor='none',
                marker='o',
                ms=ms,
                label=l,
                zorder=2.3,
            )[0]
            for l in np.unique(labels)
        ]

        # Initialize velocity arrows
        self.get_arrow_xyz_uvw = lambda frame: (self.states_3d[frame, :, :3], self.states_3d[frame, :, self.dim_3d:self.dim_3d+3])
        self.arrows = self.ax.quiver(
            [], [], [],
            [], [], [],
            arrow_length_ratio=0,
            length=arrow_length_scale,
            lw=lw,
            color='gray',
            alpha=.4,
            zorder=2.2,
        )

        # Initialize modal lines
        self.modal_lines = [
            mp3d.art3d.Line3DCollection(
                [],
                label=f'Modality {i}',
                lw=lw,
                zorder=2.1,
            )
            for i, dist in enumerate(self.modal_dist)
        ]
        for ml in self.modal_lines: self.ax.add_collection(ml)

        # Limits
        self.ax.set(
            xlim=(self.states_3d[present][:, 0].min(), self.states_3d[present][:, 0].max()),
            ylim=(self.states_3d[present][:, 1].min(), self.states_3d[present][:, 1].max()),
            zlim=(self.states_3d[present][:, 2].min(), self.states_3d[present][:, 2].max()),
        )

        # Legends
        l1 = self.ax.legend(handles=self.nodes, loc='upper left')
        self.ax.add_artist(l1)
        l2 = self.ax.legend(handles=[
            self.ax.plot([], [], color='red', markeredgecolor='none', label='Repulsion')[0],
            self.ax.plot([], [], color='blue', markeredgecolor='none', label='Attraction')[0],
        ], loc='upper right')
        self.ax.add_artist(l2)

        # Styling
        self.ax.set(xlabel='x', ylabel='y', zlabel='z')
        self.get_angle = lambda frame: (30, (360*rotations_per_second)*(frame*interval/1000)-60, 0)
        self.ax.view_init(*self.get_angle(0))

    def update(self, frame):
        super().update(frame)

        # Adjust nodes
        for i, l in enumerate(np.unique(self.labels)):
            present_labels = self.present[frame] * torch.tensor(self.labels==l)
            data = self.states_3d[frame, present_labels, :3].T.numpy()
            self.nodes[i].set_data(*data[:2])
            self.nodes[i].set_3d_properties(data[2])

        # Adjust arrows
        xyz_xyz = [[xyz, xyz+self.arrow_length_scale*uvw] for i, (xyz, uvw) in enumerate(zip(*self.get_arrow_xyz_uvw(frame))) if self.present[frame, i]]
        self.arrows.set_segments(xyz_xyz)

        # Adjust lines
        # NOTE: Currently calculates invisible lines unoptimally
        for i, (dist, ml, li) in enumerate(zip(self.modal_dist, self.modal_lines, self.line_indices)):
            ml.set_segments(self.states_3d[frame, li, :3])

            # Calculate discrepancy
            latent_dist = _utility.distance.partition_distance(self.states_3d[frame, :, :self.dim], partitions=self.partitions)
            dd_array = latent_dist[li[:, 0], li[:, 1]] - dist[li[:, 0], li[:, 1]]
            if self.partitions is not None: dd_array = dd_array.A1

            # Assign colors
            idx = self.line_indices[i].T
            rgba = self.get_rgba_from_dd_array(
                dd_array,
                [self.present[frame, li[j]].all() for j in range(li.shape[0])],
                max_value_alpha=2*dist[idx[0], idx[1]],
            )
            ml.set_color(rgba)

        # Styling
        self.ax.set_title(f'{self.skip*frame: 4} : {self.rewards[frame, self.present[frame]].mean():5.2f}')  
        self.ax.view_init(*self.get_angle(frame))


class ViewSilhouette(ViewBase):
    def __init__(
        self,
        # Data
        states,
        labels,
        *args,
        # Data params
        dim,
        # Arguments
        # None
        # Styling
        # None
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Data
        self.states = states
        self.labels = labels

        # TODO: Update from 3 to env.dim
        self.get_silhouette_samples = lambda frame: sklearn.metrics.silhouette_samples(self.states[frame, :, :dim].cpu(), self.labels)
        self.bars = [self.ax.bar(l, 0) for l in np.unique(self.labels)]

        # Styling
        self.ax.axhline(y=0, color='black')
        self.ax.set(ylim=(-1, 1))
        self.ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    
    def update(self, frame):
        super().update(frame)

        # Update barplots
        for bar, l in zip(self.bars, np.unique(self.labels)):
            bar[0].set_height(self.get_silhouette_samples(frame)[self.labels==l].mean())

        # Styling
        self.ax.set_title('Silhouette Coefficient')
        self.ax.set_xlabel('Group')
        self.ax.set_ylabel(f'Mean: {self.get_silhouette_samples(frame).mean():5.2f}') 


class ViewTemporalDiscrepancy(ViewModalDistBase):
    def __init__(
        self,
        # Data
        present,
        states,
        stages,
        modalities,
        *args,
        # Data params
        temporal_stages,
        modal_targets,
        # Arguments
        # None
        # Styling
        y_bound=[.1, np.inf],  # Bounds for discrepancy chart max
        clip_discrepancy=False,  # Clips discrepancy values to inside the chart
        dynamic_ylim=True,  # Change ylim of plot dynamically
        ylim_padding=.05,  # Percentage padding added to top ylim for cleaner presentation
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Data
        self.present = present
        self.states = states
        self.stages = stages
        self.modalities = modalities

        # Data params
        self.temporal_stages = temporal_stages
        self.modal_targets = modal_targets if modal_targets is not None else np.arange(len(self.modalities))

        # Styling
        self.y_bound = y_bound
        self.clip_discrepancy = clip_discrepancy
        self.dynamic_ylim = dynamic_ylim
        self.ylim_padding = ylim_padding

        # Initialize plot
        self.temporal_eval_plot = self.ax.plot([], [], color='black', marker='o')[0]
        # TODO: Highlight training regions
 
        # Styling
        self.ax.set_xticks(
            np.arange(len(self.temporal_stages)),
            [', '.join([str(s) for s in stage]) for stage in self.temporal_stages],
        )
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=45, ha='center', va='baseline')
        max_height = max([l.get_window_extent(renderer=self.ax.figure.canvas.get_renderer()).height for l in self.ax.get_xticklabels()])
        fontsize = self.ax.get_xticklabels()[0].get_size()
        pad = fontsize / 2 + max_height / 2
        self.ax.tick_params(axis='x', pad=pad)
        self.ax.set_xlim([-.5, len(self.temporal_stages)-.5])
        self.ax.set_ylim([0, 1]); self.current_y_max = 0
        self.ax.set_title('Temporal Discrepancy')
        self.ax.set_xlabel('Stage')
        self.ax.set_ylabel('Mean Discrepancy')
        # ax2.set_yscale('symlog')
        self.ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    def update(self, frame):
        super().update(frame)

        # Calculate discrepancy
        pos_dist = _utility.distance.euclidean_distance(self.states[frame, self.present[frame], :])
        running = torch.zeros(self.present[frame].sum())
        for dist in self.modal_dist:
            square_ew = (pos_dist - dist[self.present[frame], self.present[frame]])**2
            mean_square_ew = square_ew.mean(dim=1)
            running = running + mean_square_ew
        running = running / len(self.modal_dist)
        discrepancy = running.detach().cpu().mean()

        # Clip if needed
        y_max = np.clip(discrepancy, *self.y_bound)
        if self.clip_discrepancy: discrepancy = y_max

        # Adjust plot
        xdata = self.temporal_eval_plot.get_xdata()
        ydata = self.temporal_eval_plot.get_ydata()
        if not ((frame == 0 and len(xdata) > 0)):  # matplotlib sometimes runs frame 0 multiple times
            if frame == 0 or (self.stages[frame] != self.stages[frame-1]):
                xdata = np.append(xdata, self.stages[frame])
                ydata = np.append(ydata, None)
            # Update max discrepancy
            if (frame < self.stages.shape[0] - 1) and (self.stages[frame] != self.stages[frame+1]):
                self.current_y_max = max(self.current_y_max, discrepancy)
            ydata[-1] = discrepancy
            self.temporal_eval_plot.set_xdata(xdata)
            self.temporal_eval_plot.set_ydata(ydata)
        
        # Styling
        if self.dynamic_ylim:
            new_ylim = max(y_max, self.current_y_max)
            new_ylim = new_ylim + self.ylim_padding * new_ylim
            self.ax.set_ylim([0, new_ylim])


class ViewTemporalScatter(ViewLinesBase):
    def __init__(
        self,
        # Data
        present,
        *args,
        # Data params
        dim,
        # Arguments
        modal_targets,
        # Styling
        scaling_approach='limit',  # 'limit' y-values to top or 'scale' plot to show all
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Data
        self.present = present

        # Data params
        self.dim = dim

        # Arguments
        self.modal_targets = modal_targets  # A bit strict to require this

        # Styling
        self.scaling_approach = scaling_approach

        # Initialize plot
        # TODO: Remove outline from points
        self.points = [[
            self.ax.plot(
                [], [],
                color='black',
                markeredgecolor='none',
                marker=['o', '^', 's', 'p', 'h'][modal_num % 5],
                linestyle='',
            )[0]
            for _ in range(self.line_indices[modal_num].shape[0])
        ] for modal_num in range(len(self.modal_targets))]

        # Legends
        l1 = self.ax.legend(handles=[
            self.ax.plot(
                [], [],
                color='black',
                markeredgecolor='none',
                marker=['o', '^', 's', 'p', 'h'][modal_num % 5],
                linestyle='',
                label=f'Modality {self.modal_targets[modal_num]}',
            )[0]
            for modal_num in range(len(self.modal_targets))
        ], loc='upper left')
        self.ax.add_artist(l1)
        l2 = self.ax.legend(handles=[
            matplotlib.patches.Patch(color='red', label='Repulsion'),
            matplotlib.patches.Patch(color='blue', label='Attraction'),
        ], loc='upper right')
        self.ax.add_artist(l2)

        # Stylize
        self.ax.spines[['right', 'top']].set_visible(False)
        self.top_lim = max([md.max() for md in self.modal_dist])
        bot_top_lim = [0, self.top_lim]
        self.ax.set_xlim(bot_top_lim)
        self.ax.set_ylim(bot_top_lim)

        # Plot y=x
        self.ax.plot(bot_top_lim, bot_top_lim, 'k-', alpha=.75, zorder=0)
        if self.scaling_approach == 'limit': self.ax.set_aspect('equal')

        # Titles
        self.ax.set_title(f'Inter-Cell Distance Comparison')
        self.ax.set_xlabel('Measured')
        self.ax.set_ylabel('Predicted')

    def update(self, frame):
        super().update(frame)

        # Update positions and color
        latent_dist_total = _utility.distance.partition_distance(self.states[frame, :, :self.dim])
        for modal_num in range(len(self.modal_targets)):
            for i, idx in enumerate(self.line_indices[modal_num]):
                point = self.points[modal_num][i]

                # Show point and adjust color if present
                if np.array([self.present[frame, j] for j in idx]).all():
                    actual_dist = self.modal_dist[modal_num][idx[0], idx[1]]
                    latent_dist = latent_dist_total[idx[0], idx[1]]
                    # Limit height
                    if self.scaling_approach == 'limit': latent_dist = min(self.top_lim, latent_dist)
                    # Set position
                    point.set_data([actual_dist], [latent_dist])
                    # Set color
                    dd = latent_dist - actual_dist
                    rgba = self.get_rgba_from_dd_array([dd], min_alpha=.1, max_value_alpha=2*actual_dist)[0]
                    # Clip lowest alpha
                    point.set_color(rgba)

                # Hide point if both nodes not present
                else:
                    point.set_data([], [])

        # Update axis scaling
        if self.scaling_approach == 'scale':
            new_top_lim = latent_dist_total[self.present[frame], self.present[frame]].max()
            new_top_lim = np.ceil(new_top_lim / self.top_lim) * self.top_lim
            self.ax.set_ylim(top=new_top_lim)


class ViewPerturbationEffect(ViewModalDistBase):
    def __init__(
        self,
        # Data
        present,
        states,
        stages,
        modalities,
        *args,
        # Data params
        dim,
        perturbation_features,
        perturbation_feature_names=None,
        # Arguments
        # None
        # Styling
        default_ylim=1e-2,
        **kwargs,
    ):
        local_vars = locals().copy()
        for k in ('self', '__class__', 'args', 'kwargs'): local_vars.pop(k)
        super().__init__(**local_vars, **kwargs)

        # Data
        self.present = present
        self.states = states
        self.stages = stages

        # Data params
        self.dim = dim
        self.perturbation_features = perturbation_features
        self.perturbation_feature_names = perturbation_feature_names
        if self.perturbation_feature_names is None: self.perturbation_feature_names = [np.arange(len(pfs)) for pfs in self.perturbation_features]

        # Styling
        self.default_ylim = default_ylim

        # Get baseline for steady state
        self.steady_state = self.states[torch.argwhere(self.stages==0).max(), :, :].clone()

        # Initialize bars
        self.bars = [self.ax.bar(l, 0, color='gray') for l in range(sum([len(fs) for fs in self.perturbation_features]))]

        # Styling
        self.ax.set_xlabel('Feature')
        self.ax.set_ylabel('Effect Size')
        self.ax.set_xticks(list(range(sum([len(pfs) for pfs in self.perturbation_features]))))
        self.ax.set_xticklabels([pfn for pfns in self.perturbation_feature_names for pfn in pfns], rotation=45, ha='center', va='baseline')

        max_height = max([l.get_window_extent(renderer=self.ax.figure.canvas.get_renderer()).height for l in self.ax.get_xticklabels()])
        fontsize = self.ax.get_xticklabels()[0].get_size()
        pad = fontsize / 2 + max_height / 2
        self.ax.tick_params(axis='x', pad=pad)

        # Additional styling
        self.ax.set_ylim([0, self.default_ylim])
        self.ax.spines[['right', 'top', 'left']].set_visible(False)
        formatter = ScalarFormatter()
        formatter.set_powerlimits((0, 0))
        self.ax.yaxis.set_major_formatter(formatter)

    def update(self, frame):
        super().update(frame)

        # Calclate mean positional difference from steady state (effect size)
        diff = (self.states[frame, self.present[frame], :self.dim] - self.steady_state[self.present[frame], :self.dim]).square().sum(dim=-1).sqrt().mean(dim=-1)

        # Reset ylim for passing integration stage
        if frame == torch.argwhere(self.stages==0).max() + 1:
            self.ax.set_ylim([0, self.default_ylim])

        # Set bar heights
        stage = self.stages[frame]
        # Set all bar heights for integration period
        if stage == 0:
            for bar in self.bars: bar[0].set_height(diff)
        # Set individual bar heights for perturbation effects
        else: self.bars[stage - 1][0].set_height(diff)

        # Set new limit
        ylim = self.ax.get_ylim()
        self.ax.set_ylim([0, max(diff, ylim[1])])