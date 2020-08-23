import causal_world.evaluation.visualization.visualiser as vis
from causal_world.evaluation.visualization.utils import data_array_from_aggregated_experiments
from causal_world.evaluation.visualization.bar_plots import bar_plots_with_protocol_table

import argparse
import json
import os
import numpy as np


def get_mean_scores(scores_list):
    scores_mean = dict()
    num_scores = len(scores_list)
    for key in list(scores_list[0].keys())[:-2]:
        scores_mean[key] = {}
        for sub_key in scores_list[0][key].keys():
            scores_mean[key][sub_key] = np.mean(
                [scores_list[i][key][sub_key] for i in range(num_scores)])
            scores_mean[key][sub_key + '_std'] = np.std(
                [scores_list[i][key][sub_key] for i in range(num_scores)])
    return scores_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument("--title", required=True, help="title")

    args = vars(parser.parse_args())
    output_path = str(args['output_path'])
    title = str(args['title'])

    # PPO_scores_path = 'baselines_picking/5/evaluation/scores.json'
    # SAC_scores_path = 'baselines_picking/20/evaluation/scores.json'
    # TD3_scores_path = 'baselines_picking/35/evaluation/scores.json'
    NUM_RANDOM_SEEDS = 5
    num_curriculum = 3
    algorithm_list = ['PPO', 'SAC', 'TD3']

    experiments = dict()

    plotting_path = os.path.join(output_path, 'baseline_dummy')
    for algorithm_index, algorithm in enumerate(algorithm_list):
        for curriculum in range(num_curriculum):
            label = algorithm + '(' + str(curriculum) + ')'
            scores_list = list()
            for random_seed in range(NUM_RANDOM_SEEDS):
                model_num = algorithm_index * (num_curriculum * NUM_RANDOM_SEEDS) \
                            + curriculum * NUM_RANDOM_SEEDS \
                            + random_seed
                scores_path = os.path.join(output_path, str(model_num),
                                           'evaluation', 'scores.json')
                if os.path.exists(scores_path):
                    with open(scores_path, 'r') as fin:
                        scores = json.load(fin)
                        scores_list.append(scores)
            if len(scores_list) > 0:
                aggregated_scores = get_mean_scores(scores_list)
                experiments[label] = aggregated_scores

    if os.path.exists('protocol_settings.json'):
        with open('protocol_settings_2.json', 'r') as fin:
            protocol_settings = json.load(fin)
    data = data_array_from_aggregated_experiments(experiments)
    bar_plots_with_protocol_table(output_path, data, protocol_settings, title)
