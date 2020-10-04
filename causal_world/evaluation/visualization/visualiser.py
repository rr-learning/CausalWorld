from causal_world.evaluation.visualization.radar_plots import radar_plots
from causal_world.evaluation.visualization.bar_plots import bar_plots
from causal_world.evaluation.visualization.utils import aggregated_data_from_experiments

import os


def generate_visual_analysis(output_path, experiments):
    """
    saves bar plots as well as radar plots for quick comparisons of the
    policies passed.

    :param output_path: (str) specifies the output path for saving the plot
                              results.
    :param experiments: (dict) specifies the experiment name as a key and the
                               scores json data as the value.
    :return:
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data = aggregated_data_from_experiments(experiments)

    radar_plots(output_path, data)
    bar_plots(output_path, data)
    return
