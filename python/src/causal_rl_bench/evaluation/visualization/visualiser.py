from causal_rl_bench.evaluation.visualization.radar_plots import radar_plots

import os


def generate_visual_analysis(output_path, experiments):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    radar_plots(output_path, experiments)
