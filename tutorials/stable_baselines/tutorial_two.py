from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.causalworld import CausalWorld
from stable_baselines import HER, SAC
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import json
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from causal_rl_bench.wrappers.env_wrappers import HERGoalEnvWrapper
import argparse


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, sac_config, total_time_steps,
                 validate_every_timesteps, task_name):
    def _make_env(rank):
        def _init():
            task = task_generator(task_generator_id=task_name)
            env = CausalWorld(task=task, skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank, max_episode_length=
                        maximum_episode_length)
            env = HERGoalEnvWrapper(env)
            return env

        set_global_seeds(seed_num)
        return _init
    os.makedirs(log_relative_path)
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = HER('MlpPolicy', env, SAC,
                verbose=1, policy_kwargs=dict(layers=[256, 256, 256]),
                **sac_config)
    save_config_file(sac_config, _make_env(0)(),
                     os.path.join(log_relative_path, 'config.json'))
    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name="sac",
                    reset_num_timesteps=False)
    model.save(os.path.join(log_relative_path, 'saved_model'))
    return


def save_config_file(sac_config, env, file_path):
    task_config = env.get_task().get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, sac_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #TODO: pass reward weights here!!
    ap.add_argument("--seed_num", required=False, default=0,
                    help="seed number")
    ap.add_argument("--skip_frame", required=False, default=10,
                    help="skip frame")
    ap.add_argument("--max_episode_length", required=False, default=2500,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update", required=False,
                    default=150000, help="total time steps per update")
    ap.add_argument("--task_name", required=False,
                    default="reaching", help="the task nam for training")
    ap.add_argument("--log_relative_path", required=True,
                    help="log folder")
    args = vars(ap.parse_args())
    total_time_steps_per_update = int(args['total_time_steps_per_update'])
    num_of_envs = 1
    log_relative_path = str(args['log_relative_path'])
    maximum_episode_length = int(args['max_episode_length'])
    skip_frame = int(args['skip_frame'])
    seed_num = int(args['seed_num'])
    task_name = str(args['task_name'])
    assert (((float(total_time_steps_per_update) /
             num_of_envs)/5).is_integer())
    #params are taken from
    #https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/her.yml
    sac_config = {"n_sampled_goal": 4,
                  "goal_selection_strategy": 'future',
                  "learning_rate": 0.001,
                  "train_freq": 1,
                  "gradient_steps": 1,
                  "learning_starts": 1000,
                  "tensorboard_log": log_relative_path,
                  "buffer_size": int(1e6),
                   "gamma": 0.95,
                   "batch_size": 256,
                   "ent_coef": 'auto',
                   "random_exploration": 0.1}
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 sac_config=sac_config,
                 total_time_steps=60000000,
                 validate_every_timesteps=1000000,
                 task_name=task_name)
