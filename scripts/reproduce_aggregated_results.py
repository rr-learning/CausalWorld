from stable_baselines import TD3, PPO2, SAC
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
from causal_rl_bench.benchmark.benchmarks import REACHING_BENCHMARK, \
    PUSHING_BENCHMARK, \
    PICKING_BENCHMARK, \
    PICK_AND_PLACE_BENCHMARK, \
    TOWER_2_BENCHMARK
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_actors import RandomInterventionActorPolicy, GoalInterventionActorPolicy
import argparse
import os
import json
import numpy as np

world_seed = 0
num_of_envs = 4

NUM_RANDOM_SEEDS = 5
NET_LAYERS = [256, 256]


def load_model_settings(file_path):
    with open(file_path, 'r') as fin:
        model_settings = json.load(fin)
    return model_settings


def load_model_from_settings(model_settings, model_path, time_steps):
    algorithm = model_settings['algorithm']
    model = None
    policy_path = os.path.join(model_path,
                               'model_' + str(time_steps) + '_steps')
    if algorithm == 'PPO':
        model = PPO2.load(policy_path)
    elif algorithm == 'SAC':
        model = SAC.load(policy_path)
    elif algorithm == 'TD3':
        model = TD3.load(policy_path)
    return model


def protocols_from_settings(model_settings):
    task_generator_id = model_settings['benchmarks']['task_generator_id']
    protocols = None
    if task_generator_id == 'pushing':
        protocols = PUSHING_BENCHMARK
    elif task_generator_id == 'reaching':
        protocols = REACHING_BENCHMARK
    elif task_generator_id == 'picking':
        protocols = PICKING_BENCHMARK
    elif task_generator_id == 'pick_and_place':
        protocols = PICK_AND_PLACE_BENCHMARK
    elif task_generator_id == 'towers':
        protocols = TOWER_2_BENCHMARK
    return protocols['evaluation_protocols']


def get_mean_scores(scores_list):
    scores_mean = dict()
    num_scores = len(scores_list)
    for key in scores_list[0].keys():
        scores_mean[key] = {}
        for sub_key in scores_list[0][key].keys():
            scores_mean[key][sub_key] = np.mean(
                [scores_list[i][key][sub_key] for i in range(num_scores)])
    return scores_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument("--models_path", required=True, help="models path")
    parser.add_argument("--benchmark", required=True, help="models path")

    args = vars(parser.parse_args())
    output_path = str(args['output_path'])
    models_path = str(args['models_path'])
    benchmark = str(args['benchmark'])

    algorithm_list = ['PPO', 'SAC', 'TD3']

    num_algorithms = len(algorithm_list)

    training_curriculum = [
        'no_intervention', 'goal_intervention', 'random_intervention'
    ]
    num_curriculum = len(training_curriculum)

    experiments = dict()

    plotting_path = os.path.join(output_path, benchmark)
    for algorithm_index, algorithm in enumerate(algorithm_list):
        for curriculum_index, curriculum in enumerate(training_curriculum):
            label = algorithm + ' + ' + curriculum
            scores_list = []
            for random_seed in range(NUM_RANDOM_SEEDS):
                model_num = algorithm_index * (num_curriculum * NUM_RANDOM_SEEDS) \
                            + curriculum_index * NUM_RANDOM_SEEDS \
                            + random_seed
                scores_path = os.path.join(models_path, str(model_num),
                                           'evaluation', 'scores.json')
                if os.path.exists(scores_path):
                    with open(scores_path, 'r') as fin:
                        scores = json.load(fin)
                        scores_list.append(scores)
            if len(scores_list) > 0:
                mean_scores = get_mean_scores(scores_list)
                experiments[label] = mean_scores
    if bool(experiments):
        if not os.path.exists(plotting_path):
            os.makedirs(plotting_path)
        vis.generate_visual_analysis(plotting_path, experiments=experiments)
