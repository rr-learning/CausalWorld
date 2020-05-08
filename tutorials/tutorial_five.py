import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task

import gym
from gym import spaces
import numpy as np

seed = 0

import matplotlib.pyplot as plt


class CameraObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, scaling_factor=10, single_camera=True):
        gym.ObservationWrapper.__init__(self, env)
        height = int(540 / scaling_factor)
        width = int(720 / scaling_factor)
        self.scaling_factor = scaling_factor
        self.single_camera = single_camera
        height_multiplier = 3
        if self.single_camera:
            height_multiplier = 1
        if self.env.robot.normalize_observations:
            self.observation_space = spaces.Box(low=-np.ones(shape=(height*height_multiplier, width, 3)),
                                                high=np.ones(shape=(height*height_multiplier, width, 3)),
                                                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.zeros(shape=(height*height_multiplier, width, 3)),
                                                high=np.full(shape=(height*height_multiplier, width, 3), fill_value=255),
                                                dtype=np.uint8)

    def observation(self, observation):
        if self.single_camera:
            return observation[1, ::self.scaling_factor, ::self.scaling_factor,:]
        else:
            return np.concatenate(observation[:, ::self.scaling_factor, ::self.scaling_factor, :])


def _make_env(rank):
    def _init():
        task = Task(task_id='pushing')
        env = World(task=task, skip_frame=20,
                    enable_visualization=False,
                    observation_mode='cameras',
                    normalize_observations=False,
                    seed=seed + rank)
        env.enforce_max_episode_length(episode_length=50)
        env = CameraObservationWrapper(env, single_camera=True)
        return env
    set_global_seeds(seed)
    return _init


def test_observation_wrapper():
    task = Task(task_id='pushing')
    env = World(task=task, skip_frame=20,
                enable_visualization=False,
                observation_mode='cameras',
                normalize_observations=True,
                seed=0)
    env.enforce_max_episode_length(episode_length=50)
    env = CameraObservationWrapper(env, single_camera=False)

    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
    env.reset()


def train_policy(num_of_envs):
    total_time_steps = 200000000
    validate_every_timesteps = 1000000

    # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO2(CnnPolicy, env, gamma=0.9988, n_steps=int(200000 / num_of_envs),
                 ent_coef=0,
                 learning_rate=0.001, vf_coef=0.99,
                 max_grad_norm=0.1, lam=0.95, nminibatches=5,
                 noptepochs=100, cliprange=0.2,
                 _init_setup_model=True,
                 verbose=1,
                 tensorboard_log='output/logs_cnn_sc')
    for i in range(int(total_time_steps/validate_every_timesteps)):

        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name="ppo2_simple_reward",
                    reset_num_timesteps=False)
        model.save('output/models/pushing_model_cnn_sc_{}'.format(i * validate_every_timesteps))
    return model


if __name__ == '__main__':
    train_policy(40)

