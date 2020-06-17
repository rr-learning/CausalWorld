import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import TD3, PPO2, SAC
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
import argparse
import os
import json
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_agents import RandomInterventionActorPolicy, GoalInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.benchmark.benchmarks import PUSHING_BENCHMARK
from stable_baselines.ddpg.noise import NormalActionNoise

world_seed = 0
num_of_envs = 20


def save_config_file(file_path, world_params, task_configs):
    configs_to_save = {'task_params': task_configs,
                       'world_params': world_params}
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


def baseline_model(model_num):
    task_generator_ids = ['reaching',
                          'pushing',
                          'picking',
                          'pick_and_place',
                          'towers']
    random_seeds = np.arange(start=0, stop=5)
    algorithms = ['PPO',
                  'SAC',
                  'TD3',
                  'DDPG_HER']
    curriculum_kwargs_1 = {'intervention_actors': [],
                           'actives': []}
    curriculum_kwargs_2 = {'intervention_actors': [GoalInterventionActorPolicy()],
                           'actives': [(0, 1e9, 5, 0)]}
    curriculum_kwargs_3 = {'intervention_actors': [RandomInterventionActorPolicy()],
                           'actives': [(0, 1e9, 5, 0)]}
    curriculum_kwargs = [curriculum_kwargs_1,
                         curriculum_kwargs_2,
                         curriculum_kwargs_3]

    return sweep_product([task_generator_ids,
                          algorithms,
                          curriculum_kwargs,
                          random_seeds])[model_num]


def sweep_product(list_of_settings):
    if len(list_of_settings) == 1:
        return list_of_settings[0]
    result = []
    other_items = sweep_product(list_of_settings[1:])
    for first_dict in list_of_settings[0]:
        for second_dict in other_items:
            new_dict = {}
            new_dict.update(first_dict)
            new_dict.update(second_dict)
            result.append(new_dict)
    return result


def get_single_process_env(model_settings):
    task = task_generator(**model_settings['task_configs'])
    env = World(task=task,
                **model_settings['world_params'],
                seed=world_seed,
                max_episode_length=task.get_max_episode_length())
    env = CurriculumWrapper(env,
                            intervention_actors=model_settings["intervention_actors"],
                            actives=model_settings["actives"])
    return env


def get_multi_process_env(model_settings):
    def _make_env(rank):
        def _init():
            task = task_generator(**model_settings['task_configs'])
            env = World(task=task,
                        **model_settings['world_params'],
                        seed=world_seed + rank)
            env = CurriculumWrapper(env,
                                    intervention_actors=model_settings["intervention_actors"],
                                    actives=model_settings["actives"])
            return env

        set_global_seeds(world_seed)
        return _init
    return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])


def get_TD3_model(model_settings, model_path):
    env = get_single_process_env(model_settings)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    model = TD3(TD3MlpPolicy, env, action_noise=action_noise, _init_setup_model=True,
                verbose=1, tensorboard_log=model_path,
                policy_kwargs=policy_kwargs,
                seed=model_settings['seed'])
    return model


def get_SAC_model(model_settings, model_path):
    env = get_single_process_env(model_settings)
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256])
    model = SAC(SACMlpPolicy, env, _init_setup_model=True,
                verbose=1, tensorboard_log=model_path,
                policy_kwargs=policy_kwargs,
                seed=model_settings['seed'])
    return model


def get_PPO_model(model_settings, model_path):
    env = get_multi_process_env(model_settings)
    ppo_config = {"gamma": 0.99,
                  "n_steps": 600,
                  "ent_coef": 0.01,
                  "learning_rate": 0.00025,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "nminibatches": 4,
                  "noptepochs": 4}
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    model = PPO2(MlpPolicy, env, _init_setup_model=True, policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config, tensorboard_log=model_path,
                 seed=model_settings['seed'])
    return model


def train_model_num(model_settings, output_path):
    total_time_steps = 1000000
    validate_every_timesteps = 500000
    model_path = os.path.join(output_path, 'model')
    os.makedirs(model_path)
    if model_settings['algorithm'] == 'PPO':
        model, env = get_PPO_model(model_settings, model_path)
        num_of_active_envs = num_of_envs
    elif model_settings['algorithm'] == 'SAC':
        model, env = get_SAC_model(model_settings, model_path)
        num_of_active_envs = 1
    elif model_settings['algorithm'] == 'TD3':
        model, env = get_TD3_model(model_settings, model_path)
        num_of_active_envs = 1
    else:
        model, env = get_PPO_model(model_settings, model_path)
        num_of_active_envs = num_of_envs

    save_config_file(os.path.join(model_path, 'config.json'), world_params, task_configs)

    checkpoint_callback = CheckpointCallback(save_freq=int(validate_every_timesteps / num_of_active_envs),
                                             save_path=model_path,
                                             name_prefix='model')

    model.learn(int(total_time_steps / num_of_active_envs), callback=checkpoint_callback)
    env.save_world(output_path)
    env.close()

    return model


def get_configs(model_num):
    world_params = {'skip_frame': 3,
                    'enable_visualization': True,
                    'observation_mode': 'structured',
                    'normalize_observations': True,
                    'enable_goal_image': False,
                    'action_mode': 'joint_positions',
                    'max_episode_length': 600}

    task_configs = {'task_generator_id': 'pushing',
                    'intervention_split': True,
                    'training': True,
                    'sparse_reward_weight': 1,
                    'dense_reward_weights': [10, 1, 0.5]}


    return task_configs, world_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", required=True, default=0,
                        help="model number")
    parser.add_argument("--output_path", required=True,
                        help="output path")

    args = vars(parser.parse_args())
    model_num = int(args['model_num'])
    output_path = str(args['output_path'])

    output_path = os.path.join(output_path, str(model_num))
    os.makedirs(output_path)

    model_settings = baseline_model(model_num)

    model = train_model_num(model_settings, output_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]


    animation_path = os.path.join(output_path, 'animation')
    os.makedirs(animation_path)
    # Record a video of the policy is done in one line
    viewer.record_video_of_policy(task=task_generator(**task_configs),
                                  world_params=world_params,
                                  policy_fn=policy_fn,
                                  file_name=os.path.join(animation_path, "pushing_video"),
                                  number_of_resets=1,
                                  max_time_steps=600)
    evaluation_path = os.path.join(output_path, 'evaluation')
    os.makedirs(evaluation_path)

    default_evaluation_protocols = PUSHING_BENCHMARK['evaluation_protocols']

    evaluator = EvaluationPipeline(evaluation_protocols=
                                   default_evaluation_protocols,
                                   tracker_path=output_path,
                                   intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    evaluator.save_scores(evaluation_path)
    experiments = dict()
    experiments['TD3_default'] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
    print("success")
