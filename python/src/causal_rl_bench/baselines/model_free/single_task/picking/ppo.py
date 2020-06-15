from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
import numpy as np


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name):
    def _make_env(rank):
        def _init():
            task = task_generator(task_generator_id=task_name,
                                  dense_reward_weights=np.array([250, 0, 125,
                                                                 0, 750, 0, 0,
                                                                 0.005]),
                                  sparse_reward_weight=0,
                                  goal_height=0.15,
                                  tool_block_mass=0.02)
            env = World(task=task, skip_frame=skip_frame,
                        enable_visualization=False,
                        seed=seed_num + rank, max_episode_length=
                        maximum_episode_length)
            return env

        set_global_seeds(seed_num)
        return _init
    os.makedirs(log_relative_path)
    checkpoint_callback = CheckpointCallback(save_freq=validate_every_timesteps,
                                             save_path=log_relative_path,
                                             name_prefix='model')
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO2(MlpPolicy, env, _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config)
    model.learn(total_timesteps=validate_every_timesteps,
                tb_log_name="ppo2",
                reset_num_timesteps=False,
                callback=checkpoint_callback)
    return


total_time_steps_per_update = 1000000
num_of_envs = 20
log_relative_path = 'baseline_picking_ppo'
maximum_episode_length = 600
skip_frame = 3
seed_num = 0
task_name = 'picking'
ppo_config = {"gamma": 0.99,
              "n_steps": 600,
              "ent_coef": 0.01,
              "learning_rate": 0.00025,
              "vf_coef": 0.5,
              "max_grad_norm": 0.5,
              "nminibatches": 4,
              "noptepochs": 4,
              "tensorboard_log": log_relative_path}
train_policy(num_of_envs=num_of_envs,
             log_relative_path=log_relative_path,
             maximum_episode_length=maximum_episode_length,
             skip_frame=skip_frame,
             seed_num=seed_num,
             ppo_config=ppo_config,
             total_time_steps=60000000,
             validate_every_timesteps=1000000,
             task_name=task_name)

