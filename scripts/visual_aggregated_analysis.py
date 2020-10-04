from causal_world.evaluation.visualization.utils import aggregated_data_from_experiments
from causal_world.evaluation.visualization.bar_plots import bar_plots_with_protocol_table

import argparse
import json
import os
import numpy as np

# TODO: This should be returned from the individual scripts after refactoring there.
protocol_settings = {
    "P0": {
        "space": "-",
        "variables": "-",
        "time step": 0
    },
    "P1": {
        "space": "A",
        "variables": "tm",
        "time step": 0
    },
    "P2": {
        "space": "B",
        "variables": "tm",
        "time step": 0
    },
    "P3": {
        "space": "A",
        "variables": "ts",
        "time step": 0
    },
    "P4": {
        "space": "A",
        "variables": "tp",
        "time step": 0
    },
    "P5": {
        "space": "A",
        "variables": "gp",
        "time step": 0
    },
    "P6": {
        "space": "B",
        "variables": "tp, gp",
        "time step": 0
    },
    "P7": {
        "space": "A",
        "variables": "tp, gp, \n tm",
        "time step": 0
    },
    "P8": {
        "space": "B",
        "variables": "tp, gp, \n tm",
        "time step": 0
    },
    "P9": {
        "space": "B",
        "variables": "tp, gp, \n tm, ff",
        "time step": 0
    },
    "P10": {
        "space": "A",
        "variables": "all",
        "time step": 0
    },
    "P11": {
        "space": "B",
        "variables": "all",
        "time step": 0
    }
}


def get_mean_scores(scores_list):
    scores_mean = dict()
    num_scores = len(scores_list)
    for key in list(scores_list[0].keys())[:]:
        scores_mean[key] = {}
        for sub_key in scores_list[0][key].keys():
            scores_mean[key][sub_key] = np.mean(
                [scores_list[i][key][sub_key] for i in range(num_scores)])
            scores_mean[key][sub_key + '_std'] = np.std(
                [scores_list[i][key][sub_key] for i in range(num_scores)])
    return scores_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", required=True, help="models_path")
    parser.add_argument("--title", required=True, help="title")

    args = vars(parser.parse_args())
    models_path = str(args['models_path'])
    title = str(args['title'])

    NUM_RANDOM_SEEDS = 5
    num_curriculum = 3
    algorithm_list = ['PPO', 'SAC', 'TD3']
    time_steps = ['100000000', '10000000', '10000000']

    experiments = dict()

    for algorithm_index, algorithm in enumerate(algorithm_list):
        for curriculum in range(num_curriculum):
            label = algorithm + '(' + str(curriculum) + ')'
            scores_list = list()
            for random_seed in range(NUM_RANDOM_SEEDS):
                model_num = algorithm_index * (num_curriculum * NUM_RANDOM_SEEDS) \
                            + curriculum * NUM_RANDOM_SEEDS \
                            + random_seed
                time_string = time_steps[algorithm_index]
                # This is the default path saved from complete run
                scores_path = os.path.join(models_path, str(model_num),
                                           'evaluation',
                                           'time_steps_{}'.format(time_string),
                                           'scores.json')
                if os.path.exists(scores_path):
                    with open(scores_path, 'r') as fin:
                        scores = json.load(fin)
                        scores_list.append(scores)
            if len(scores_list) > 0:
                aggregated_scores = get_mean_scores(scores_list)
                experiments[label] = aggregated_scores

    data = aggregated_data_from_experiments(experiments, contains_err=True)
    bar_plots_with_protocol_table(models_path, data, protocol_settings, title)
