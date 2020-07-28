import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

import os


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts


def get_radar_data_from_experiments(experiments):
    data = []
    evaluation_protocol_labels = list(experiments[list(
        experiments.keys())[0]].keys())
    data.append(evaluation_protocol_labels)

    # TODO infer that from dict once more metrics are added
    metric_labels = [
        'mean_full_integrated_fractional_success',
        'mean_last_integrated_fractional_success',
        'mean_last_fractional_success'
    ]
    for metric_label in metric_labels:
        metric_scores = []
        for experiment in list(experiments.keys()):
            experiment_metric_scores = []
            for evaluation_protocol in evaluation_protocol_labels:
                experiment_metric_scores.append(
                    experiments[experiment][evaluation_protocol][metric_label])
            metric_scores.append(experiment_metric_scores)
        metric_data = (metric_label, metric_scores)
        data.append(metric_data)
        data.append(metric_data)
        data.append(metric_data)
        data.append(metric_data)
    return data


def radar_plots(output_path, experiments):
    N = len(list(experiments[list(experiments.keys())[0]].keys()))
    theta = radar_factory(N, frame='circle')

    data = get_radar_data_from_experiments(experiments)
    spoke_labels = data.pop(0)

    colors = ['b', 'r', 'g', 'm', 'y']
    colors = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6'
    ]
    for (title, case_data) in data:

        fig, ax = plt.subplots(figsize=(9, 9),
                               nrows=1,
                               ncols=1,
                               subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title,
                     weight='bold',
                     size='medium',
                     position=(0.5, 1.1),
                     horizontalalignment='center',
                     verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.0)
        ax.set_varlabels(spoke_labels)
        ax.set_ylim(0, 1.0)

        # add legend relative to top-left plot
        labels = experiments.keys()
        legend = ax.legend(labels,
                           loc=(0.85, .95),
                           labelspacing=0.1,
                           fontsize='small')

        fig.text(0.5,
                 0.965,
                 'radar_plots_automatic_evaluation_causal_rl_bench',
                 horizontalalignment='center',
                 color='black',
                 weight='bold',
                 size='large')

        plt.savefig(
            os.path.join(output_path, 'radar_plots_{}.png'.format(title)))
