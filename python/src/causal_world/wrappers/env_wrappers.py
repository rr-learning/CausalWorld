import numpy as np
from gym import spaces
import gym


class HERGoalEnvWrapper(gym.GoalEnv):

    def __init__(self, env, activate_sparse_reward=False):
        """

        :param env:
        :param activate_sparse_reward:
        """
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        current_goal = self.env.get_task().get_achieved_goal().flatten()
        goal_space_shape = current_goal.shape
        #TODO: get the actual bonds here for proper normalization maybe?
        self.action_space = self.env.action_space
        if activate_sparse_reward:
            self.env.get_task().activate_sparse_reward()
        self.observation_space = spaces.Dict(
            dict(desired_goal=spaces.Box(-np.inf,
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
        self.env.add_wrapper_info({
            'her_environment': {
                'activate_sparse_reward': activate_sparse_reward
            }
        })

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        """

        :return:
        """
        return self.env.spec

    @classmethod
    def class_name(cls):
        """

        :return:
        """
        return cls.__name__

    def step(self, action):
        """

        :param action:
        :return:
        """
        obs_dict = dict()
        normal_obs, reward, done, info = self.env.step(action)
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = info['achieved_goal'].flatten()
        obs_dict['desired_goal'] = info['desired_goal'].flatten()
        return obs_dict, reward, done, info

    def reset(self):
        """

        :return:
        """
        obs_dict = dict()
        normal_obs = self.env.reset()
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = self.env.get_task().get_achieved_goal(
        ).flatten()
        obs_dict['desired_goal'] = self.env.get_task().get_desired_goal(
        ).flatten()
        return obs_dict

    def render(self, mode='human', **kwargs):
        """

        :param mode:
        :param kwargs:
        :return:
        """
        return self.env.render(mode, **kwargs)

    def close(self):
        """

        :return:
        """
        return self.env.close()

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """

        :param achieved_goal:
        :param desired_goal:
        :param info:
        :return:
        """
        return self.env.get_task().compute_reward(achieved_goal, desired_goal,
                                                  info)

    def __str__(self):
        """

        :return:
        """
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        """

        :return:
        """
        return str(self)

    @property
    def unwrapped(self):
        """

        :return:
        """
        return self.env.unwrapped
