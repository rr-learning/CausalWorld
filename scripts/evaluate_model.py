from stable_baselines import TD3, PPO2, SAC, HER
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
from causal_rl_bench.benchmark.benchmarks import REACHING_BENCHMARK, \
    PUSHING_BENCHMARK, \
    PICKING_BENCHMARK, \
    PICK_AND_PLACE_BENCHMARK, \
    TOWER_2_BENCHMARK
import causal_rl_bench.evaluation.visualization.visualiser as vis

import argparse
import os
import json


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument("--time_steps", required=True, help="time steps")

    args = vars(parser.parse_args())
    time_steps = int(args['time_steps'])
    output_path = str(args['output_path'])

    model_path = os.path.join(output_path, 'model')
    model_settings = load_model_settings(
        os.path.join(model_path, 'model_settings.json'))

    model = load_model_from_settings(model_settings, model_path, time_steps)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    evaluation_path = os.path.join(output_path, 'evaluation',
                                   'time_steps_{}'.format(time_steps))
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)

    evaluation_protocols = protocols_from_settings(model_settings)

    evaluator = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                   tracker_path=output_path,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    evaluator.save_scores(evaluation_path, prefix=str(time_steps))
    experiments = dict()
    experiments['model'] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
