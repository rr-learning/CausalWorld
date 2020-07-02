import numpy as np
from gym import spaces
import gym


class HERGoalEnvWrapper(gym.GoalEnv):
    def __init__(self, env,
                 activate_sparse_reward=False):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        current_goal = self.env.get_task().get_achieved_goal()
        goal_space_shape = current_goal.shape
        #TODO: get the actual bonds here for proper normalization maybe?
        self.action_space = self.env.action_space
        if activate_sparse_reward:
            self.env.get_task().activate_sparse_reward()
        self.observation_space = spaces.Dict(dict(desired_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype=np.float64),
                                                  achieved_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype=np.float64),
                                                  observation=self.env.observation_space))
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.env._add_wrapper_info(
            {'her_environment': {'activate_sparse_reward': activate_sparse_reward}})

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        obs_dict = dict()
        normal_obs, reward, done, info = self.env.step(action)
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = info['achieved_goal']
        obs_dict['desired_goal'] = info['desired_goal']
        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = dict()
        normal_obs = self.env.reset()
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = self.env.get_task().get_achieved_goal()
        obs_dict['desired_goal'] = self.env.get_task().get_desired_goal()
        return obs_dict

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.get_task().compute_reward(achieved_goal,
                                                  desired_goal,
                                                  info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped
