import numpy as np
from gym import spaces
import gym


class HERGoalEnvWrapper(gym.GoalEnv):

    def __init__(self, env, activate_sparse_reward=False):
        """

        :param env: (causal_world.CausalWorld) the environment to convert.
        :param activate_sparse_reward: (bool) True to activate sparse rewards.
        """
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        current_goal = self.env.get_task().get_achieved_goal().flatten()
        goal_space_shape = current_goal.shape
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
        Used to step through the enviroment.

        :param action: (nd.array) specifies which action should be taken by
                                  the robot, should follow the same action
                                  mode specified.

        :return: (nd.array) specifies the observations returned after stepping
                            through the environment. Again, it follows the
                            observation_mode specified.
        """
        obs_dict = dict()
        normal_obs, reward, done, info = self.env.step(action)
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = info['achieved_goal'].flatten()
        obs_dict['desired_goal'] = info['desired_goal'].flatten()
        return obs_dict, reward, done, info

    def reset(self):
        """
        Resets the environment to the current starting state of the environment.

        :return: (nd.array) specifies the observations returned after resetting
                            the environment. Again, it follows the
                            observation_mode specified.
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
        Returns an RGB image taken from above the platform.

        :param mode: (str) not taken in account now.

        :return: (nd.array) an RGB image taken from above the platform.
        """
        return self.env.render(mode, **kwargs)

    def close(self):
        """
        closes the environment in a safe manner should be called at the
        end of the program.

        :return: None
        """
        return self.env.close()

    def seed(self, seed=None):
        """
        Used to set the seed of the environment,
        to reproduce the same randomness.

        :param seed: (int) specifies the seed number

        :return: (int in list) the numpy seed that you can use further.
        """
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Used to calculate the reward given a hypothetical situation that could
        be used in hindsight experience replay algorithms variants.
        Can only be used in the spare reward setting for the other setting
        it can be tricky here.

        :param achieved_goal: (nd.array) specifies the achieved goal as bounding boxes of
                            objects by default.
        :param desired_goal: (nd.array) specifies the desired goal as bounding boxes of
                            goal shapes by default.
        :param info: (dict) not used for now.

        :return: (float) the final reward achieved given the hypothetical
                         situation.
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
