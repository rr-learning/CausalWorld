from collections import OrderedDict

import numpy as np
from gym import spaces

# Important: gym mixes up ordered and unordered keys
# and the Dict space may return a different order of keys that the actual one
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class HERGoalEnvWrapper(object):
    """
    A wrapper that allow to use dict observation space (coming from GoalEnv) with
    the RL algorithms.
    It assumes that all the spaces of the dict space are of the same type.
    :param env: (gym.GoalEnv)
    """

    def __init__(self, env):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        current_goal = self.env.task.get_achieved_goal()
        goal_space_shape = current_goal.shape
        #TODO: get the actual bonds here for proper normalization maybe?
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict(dict(desired_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype='float32'),
                                                  achieved_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype='float32'),
                                                  observation=self.env.observation_space))
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

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
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped