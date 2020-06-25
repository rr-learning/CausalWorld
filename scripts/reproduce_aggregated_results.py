from stable_baselines import TD3, PPO2, SAC, HER
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
from causal_rl_bench.benchmark.benchmarks import REACHING_BENCHMARK, \
    PUSHING_BENCHMARK, \
    PICKING_BENCHMARK, \
    PICK_AND_PLACE_BENCHMARK, \
    TOWER_2_BENCHMARK
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_agents import RandomInterventionActorPolicy, GoalInterventionActorPolicy
import argparse
import os
import json
import numpy as np

world_seed = 0
num_of_envs = 4

NUM_RANDOM_SEEDS = 5
NET_LAYERS = [256, 256]


def baseline_model(model_num):
    benchmarks = sweep('benchmarks', [REACHING_BENCHMARK,
                                      PUSHING_BENCHMARK,
                                      PICKING_BENCHMARK,
                                      PICK_AND_PLACE_BENCHMARK,
                                      TOWER_2_BENCHMARK])
    task_configs = [{'task_configs': {'intervention_split': True,
                                      'training': True,
                                      'sparse_reward_weight': 1}}]

    world_params = [{'world_params': {'skip_frame': 3,
                                      'enable_visualization': False,
                                      'observation_mode': 'structured',
                                      'normalize_observations': True,
                                      'action_mode': 'joint_positions'}}]

    random_seeds = sweep('seed', list(range(NUM_RANDOM_SEEDS)))
    algorithms = sweep('algorithm', ['PPO',
                                     'SAC',
                                     'TD3',
                                     'SAC_HER'])
    curriculum_kwargs_1 = {'intervention_actors': [],
                           'actives': []}
    curriculum_kwargs_2 = {'intervention_actors': [GoalInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs_3 = {'intervention_actors': [RandomInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs = [curriculum_kwargs_1,
                         curriculum_kwargs_2,
                         curriculum_kwargs_3]

    return outer_product([benchmarks,
                          world_params,
                          task_configs,
                          algorithms,
                          curriculum_kwargs,
                          random_seeds])[model_num]


def sweep(key, values):
    """Sweeps the hyperparameter across different values."""
    return [{key: value} for value in values]


def outer_product(list_of_settings):
    if len(list_of_settings) == 1:
        return list_of_settings[0]
    result = []
    other_items = outer_product(list_of_settings[1:])
    for first_dict in list_of_settings[0]:
        for second_dict in other_items:
            new_dict = {}
            new_dict.update(first_dict)
            new_dict.update(second_dict)
            result.append(new_dict)
    return result


def load_model_settings(file_path):
    with open(file_path, 'r') as fin:
        model_settings = json.load(fin)
    return model_settings


def load_model_from_settings(model_settings, model_path, time_steps):
    algorithm = model_settings['algorithm']
    model = None
    policy_path = os.path.join(model_path, 'model_' + str(time_steps) + '_steps')
    if algorithm == 'PPO':
        model = PPO2.load(policy_path)
    elif algorithm == 'SAC':
        model = SAC.load(policy_path)
    elif algorithm == 'TD3':
        model = TD3.load(policy_path)
    elif algorithm == 'SAC_HER':
        model = HER.load(policy_path)
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
            scores_mean[key][sub_key] = np.mean([scores_list[i][key][sub_key] for i in range(num_scores)])
    return scores_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="output path")
    parser.add_argument("--models_path", required=True,
                        help="models path")

    args = vars(parser.parse_args())
    output_path = str(args['output_path'])
    models_path = str(args['models_path'])

    benchmark_list = ['reaching',
                      'pushing',
                      'picking',
                      'pick_and_place',
                      'towers']

    algorithm_list = ['PPO',
                      'SAC',
                      'TD3',
                      'SAC_HER']
    num_algorithms = len(algorithm_list)

    training_curriculum = ['no_intervention',
                           'goal_intervention',
                           'random_intervention']
    num_curriculum = len(training_curriculum)


    for benchmark_index, benchmark in enumerate(benchmark_list):
        experiments = dict()
        plotting_path = os.path.join(output_path, benchmark)
        for algorithm_index, algorithm in enumerate(algorithm_list):
            for curriculum_index, curriculum in enumerate(training_curriculum):
                label = algorithm + ' + ' + curriculum
                scores_list = []
                for random_seed in range(NUM_RANDOM_SEEDS):
                    model_num = benchmark_index * (num_algorithms * num_curriculum * NUM_RANDOM_SEEDS) \
                                + algorithm_index * (num_curriculum * NUM_RANDOM_SEEDS) \
                                + num_curriculum * NUM_RANDOM_SEEDS \
                                + random_seed
                    scores_path = os.path.join(models_path, str(model_num), 'evaluation', 'scores.json')
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
