import time
import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, DDPG, GAIL, SAC
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy


class TrainerPPO(object):
    def __init__(self, seed, env):
        self.seed = seed
        self.env = env
        self.num_of_envs = 1
        return

    def train_policy(self):
        total_time_steps = 10000000
        validate_every_timesteps = 1000000
        policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
        # env = SubprocVecEnv([self._make_env(rank=i) for i in range(self.num_of_envs)])
        model = PPO2(MlpPolicy, self.env, gamma=0.9988, n_steps=int(150000 / self.num_of_envs),
                     ent_coef=0,
                     learning_rate=0.001, vf_coef=0.99,
                     max_grad_norm=0.1, lam=0.95, nminibatches=5,
                     noptepochs=100, cliprange=0.2,
                     _init_setup_model=True, policy_kwargs=policy_kwargs,
                     verbose=1,
                     tensorboard_log='./logs')
        for i in range(int(total_time_steps/validate_every_timesteps)):

            model.learn(total_timesteps=validate_every_timesteps,
                        tb_log_name="ppo2_simple_reward",
                        reset_num_timesteps=False)
            model.save('picking_model')
        return model

    def visualize_policy(self, path):
        # Load the trained agent
        self.env.enforce_max_episode_length(10000)
        model = PPO2.load(path)

        # Evaluate the agent
        # mean_reward, std_reward = evaluate_policy(model, env,
        #                                           n_eval_episodes=3)
        obs = self.env.reset()
        for i in range(5000):
            action, _ = model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
