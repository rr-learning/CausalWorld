import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine

from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

import os

import matplotlib
font = {'size': 20}

matplotlib.rc('font', **font)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.

    :param num_vars: (int) Number of variables for radar chart.
    :param frame: (str) Shape of frame surrounding axes, {'circle' | 'polygon'}.
    :return:
    """
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
    """
    Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)

    :param theta:
    :return:
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts


def radar_plots(output_path, data):
    """

    :param output_path:
    :param data:
    :return:
    """
    protocol_labels = data[0]
    experiment_labels = data[1]
    metric_labels = data[2]
    N = len(protocol_labels)
    theta = radar_factory(N, frame='circle')

    colors = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6'
    ]
    colors = ['blue', 'orange', 'green']
    for metric_label in data[3]:

        fig, ax = plt.subplots(figsize=(9, 9),
                               nrows=1,
                               ncols=1,
                               subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
        metric_scores = data[3][metric_label]
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(metric_label,
                     weight='bold',
                     size='medium',
                     position=(0.5, 1.1),
                     horizontalalignment='center',
                     verticalalignment='center')
        for experiment_label, color in zip(metric_scores, colors[:len(metric_scores)]):
            experiment_scores_mean, experiment_scores_err = metric_scores[experiment_label]
            ax.plot(theta, experiment_scores_mean, color=color)
            ax.fill(theta, experiment_scores_mean, facecolor=color, alpha=0.0)
        ax.set_varlabels(protocol_labels)
        ax.set_ylim(0, 1.0)

        # add legend relative to top-left plot
        labels = experiment_labels
        ax.legend(labels,
                  loc=(0.85, .95),
                  labelspacing=0.1,
                  fontsize='small')

        fig.text(0.5,
                 0.965,
                 'radar_plots_automatic_evaluation_causal_world',
                 horizontalalignment='center',
                 color='black',
                 weight='bold',
                 size='large')

        plt.savefig(
            os.path.join(output_path, 'radar_plots_{}.png'.format(metric_label)))
