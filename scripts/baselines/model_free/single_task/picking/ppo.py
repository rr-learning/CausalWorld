from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
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
            task = generate_task(task_generator_id=task_name,
                                  dense_reward_weights=np.array(
                                      [250, 0, 125, 0, 750, 0, 0, 0.005]),
                                  fractional_reward_weight=1,
                                  goal_height=0.15,
                                  tool_block_mass=0.02)
            env = CausalWorld(task=task,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              max_episode_length=maximum_episode_length)
            return env

        set_global_seeds(seed_num)
        return _init

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    checkpoint_callback = CheckpointCallback(save_freq=int(
        validate_every_timesteps / num_of_envs),
                                             save_path=log_relative_path,
                                             name_prefix='model')
    model = PPO2(MlpPolicy,
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="ppo2",
                callback=checkpoint_callback)
    return


if __name__ == '__main__':
    total_time_steps_per_update = 1000000
    total_time_steps = 60000000
    number_of_time_steps_per_iteration = 120000
    num_of_envs = 20
    log_relative_path = 'baseline_picking_ppo'
    maximum_episode_length = 600
    skip_frame = 3
    seed_num = 0
    task_name = 'picking'
    ppo_config = {
        "gamma": 0.99,
        "n_steps": int(number_of_time_steps_per_iteration / num_of_envs),
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "nminibatches": 40,
        "noptepochs": 4,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=total_time_steps,
                 validate_every_timesteps=total_time_steps_per_update,
                 task_name=task_name)
