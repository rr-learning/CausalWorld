from causal_world.evaluation.evaluation import EvaluationPipeline
from causal_world.intervention_actors import RandomInterventionActorPolicy, GoalInterventionActorPolicy
from causal_world.benchmark.benchmarks import PICK_AND_PLACE_BENCHMARK
from causal_world.task_generators.task import task_generator
import causal_world.viewers.task_viewer as viewer
import causal_world.evaluation.visualization.visualiser as vis

from util import utils as utils, utils_baselines as utils_baselines
import argparse
import os


def baseline_model(model_num):
    benchmarks = utils.sweep('benchmarks', [PICK_AND_PLACE_BENCHMARK])
    task_configs = [{
        'task_configs': {
            'variables_space': 'space_a',
            'fractional_reward_weight': 1,
            'dense_reward_weights': [750, 50, 250, 0, 0.005]
        }
    }]

    world_params = [{
        'world_params': {
            'skip_frame': 3,
            'enable_visualization': False,
            'observation_mode': 'structured',
            'normalize_observations': True,
            'action_mode': 'joint_positions'
        }
    }]

    net_layers = utils.sweep('NET_LAYERS', [[256, 256]])
    world_seed = utils.sweep('world_seed', [0])
    NUM_RANDOM_SEEDS = 5
    random_seeds = utils.sweep('seed', list(range(NUM_RANDOM_SEEDS)))

    ppo = {'num_of_envs': 20,
           'algorithm': 'PPO',
           'validate_every_timesteps': 2000000,
           'total_time_steps': 80000001,
           'train_configs': {
               "gamma": 0.99,
               "n_steps": int(120000 / 20),
               "ent_coef": 0.01,
               "learning_rate": 0.00025,
               "vf_coef": 0.5,
               "max_grad_norm": 0.5,
               "nminibatches": 40,
               "noptepochs": 4
           }}

    sac = {'num_of_envs': 1,
           'algorithm': 'SAC',
           'validate_every_timesteps': int(500000),
           'total_time_steps': int(4000001),
           'train_configs': {
               "gamma": 0.98,
               "tau": 0.01,
               "ent_coef": 'auto',
               "target_entropy": -9,
               "learning_rate": 0.00025,
               "buffer_size": 1000000,
               "learning_starts": 1000,
               "batch_size": 256
           }}

    td3 = {'num_of_envs': 1,
           'algorithm': 'TD3',
           'validate_every_timesteps': 500000,
           'total_time_steps': 4000001,
           'train_configs': {
               "gamma": 0.98,
               "tau": 0.01,
               "learning_rate": 0.00025,
               "buffer_size": 1000000,
               "learning_starts": 1000,
               "batch_size": 256}}

    algorithms = [ppo, sac, td3]

    curriculum_kwargs_1 = {'intervention_actors': [], 'actives': []}
    curriculum_kwargs_2 = {
        'intervention_actors': [GoalInterventionActorPolicy()],
        'actives': [(0, 1e9, 2, 0)]
    }
    curriculum_kwargs_3 = {
        'intervention_actors': [RandomInterventionActorPolicy()],
        'actives': [(0, 1e9, 2, 0)]
    }
    curriculum_kwargs = [
        curriculum_kwargs_1, curriculum_kwargs_2, curriculum_kwargs_3
    ]

    return utils.outer_product([
        benchmarks, world_params, task_configs, algorithms, curriculum_kwargs,
        random_seeds, world_seed, net_layers
    ])[model_num]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num",
                        required=True,
                        default=0,
                        help="model number")
    parser.add_argument("--output_path", required=True, help="output path")

    args = vars(parser.parse_args())
    model_num = int(args['model_num'])
    output_path = str(args['output_path'])

    output_path = os.path.join(output_path, str(model_num))
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print("Folder '{}' already exists. Will try to load existing checkpoints".format(output_path))

    model_settings = baseline_model(model_num)

    model = utils_baselines.train_model(model_settings, output_path)


    def policy_fn(obs, prev_action=None, prev_reward=None):
        return model.predict(obs, deterministic=True)[0]


    animation_path = os.path.join(output_path, 'animation')
    os.makedirs(animation_path)
    viewer.record_video_of_policy(task=task_generator(
        task_generator_id=model_settings['benchmarks']['task_generator_id'],
        **model_settings['task_configs']),
        world_params=model_settings['world_params'],
        policy_fn=policy_fn,
        file_name=os.path.join(
            animation_path, "policy"),
        number_of_resets=1,
        max_time_steps=600)
    evaluation_path = os.path.join(output_path, 'evaluation')
    os.makedirs(evaluation_path)

    evaluation_protocols = model_settings['benchmarks']['evaluation_protocols']

    evaluator = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                   tracker_path=output_path,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn, fraction=0.02)

    results_model = dict()
    results_model.update(scores)
    results_model.update(model_settings)
    results_model.update({'model_num': model_num})
    del results_model['benchmarks']
    file_name_results_model = os.path.join(evaluation_path, 'results_model.json')
    utils.save_model_settings(file_name_results_model, results_model)
    evaluator.save_scores(evaluation_path)
    experiments = dict()
    experiments[str(model_num)] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
