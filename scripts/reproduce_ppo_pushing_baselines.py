import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
import argparse
import os
import json
import gym
from gym import spaces
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.task_generators.pushing import PushingTaskGenerator
from causal_rl_bench.intervention_agents import RandomInterventionActorPolicy, GoalInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper

seed = 0


def train_policy(num_of_envs,
                 output_path,
                 world_params,
                 train_configs,
                 task_configs,
                 ppo_config,
                 total_time_steps,
                 validate_every_timesteps):
    def _make_env(rank):
        def _init():
            task = task_generator(**task_configs)
            env = World(task=task,
                        **world_params,
                        seed=seed + rank)
            env = CurriculumWrapper(env,
                                    **train_configs)
            return env

        set_global_seeds(seed)
        return _init

    model_path = os.path.join(output_path, 'model')
    os.makedirs(model_path)

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO2(MlpPolicy, env, _init_setup_model=True, policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config, tensorboard_log=model_path)

    save_config_file(os.path.join(model_path, 'config.json'), ppo_config, world_params, task_configs)

    checkpoint_callback = CheckpointCallback(save_freq=validate_every_timesteps,
                                             save_path=model_path,
                                             name_prefix='rl_model')

    model.learn(int(total_time_steps), log_interval=1000, callback=checkpoint_callback)
    env.env_method("save_world", output_path)
    env.close()
    return model


def save_config_file(file_path, ppo_config, world_params, task_configs):
    configs_to_save = {'task_params': task_configs,
                       'world_params': world_params,
                       'ppo_config': ppo_config}
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


def get_configs(model_num):
    world_params = {'skip_frame': 3,
                    'enable_visualization': False,
                    'observation_mode': 'structured',
                    'normalize_observations': True,
                    'enable_goal_image': False,
                    'action_mode': 'joint_positions',
                    'max_episode_length': 100}

    ppo_config = {"gamma": 0.99,
                  "n_steps": 600,
                  "ent_coef": 0.01,
                  "learning_rate": 0.00025,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "nminibatches": 4,
                  "noptepochs": 4}

    task_configs = {'task_generator_id': 'pushing',
                    'intervention_split': True,
                    'training': True,
                    'sparse_reward_weight': 1,
                    'dense_reward_weights': [1, 1, 1]}

    if model_num == 0:
        train_configs = {'intervention_actors': [RandomInterventionActorPolicy()],
                         'episodes_hold': [20],
                         'timesteps_hold': [None]}
    else:
        train_configs = {'intervention_actors': [GoalInterventionActorPolicy()],
                         'episodes_hold': [20],
                         'timesteps_hold': [None]}

    return task_configs, train_configs, world_params, ppo_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", required=True, default=0,
                        help="model number")
    parser.add_argument("--output_path", required=True,
                        help="output path")
    parser.add_argument("--num_of_envs", required=False,
                        default=20, help="number of parallel environments")

    args = vars(parser.parse_args())
    model_num = int(args['model_num'])
    num_of_envs = int(args['num_of_envs'])
    output_path = str(args['output_path'])

    task_configs, train_configs, world_params, ppo_config = get_configs(model_num)

    output_path = os.path.join(output_path, str(model_num))
    os.makedirs(output_path)

    model = train_policy(num_of_envs=num_of_envs,
                         output_path=output_path,
                         world_params=world_params,
                         train_configs=train_configs,
                         task_configs=task_configs,
                         ppo_config=ppo_config,
                         total_time_steps=2e8 / 2e4,
                         validate_every_timesteps=1e6 / 2e3 / num_of_envs)


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
                                  max_time_steps=100)

    evaluation_path = os.path.join(output_path, 'evaluation')
    os.makedirs(evaluation_path)

    default_evaluation_protocols = PushingTaskGenerator.get_default_evaluation_protocols()

    evaluator = EvaluationPipeline(evaluation_protocols=
                                   default_evaluation_protocols,
                                   tracker_path=output_path,
                                   intervention_split=False,
                                   visualize_evaluation=False,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    evaluator.save_scores(evaluation_path)
    experiments = dict()
    experiments['PPO_default'] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
    print("success")
