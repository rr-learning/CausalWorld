import numpy as np
from gym import spaces
import gym


class HERGoalEnvWrapper(gym.GoalEnv):
    def __init__(self, env,
                 is_goal_distance_dense=False,
                 sparse_reward_weight=1):
        super(HERGoalEnvWrapper, self).__init__()
        self.env = env
        self.metadata = self.env.metadata
        self.action_space = env.action_space
        current_goal = self.env.task.get_achieved_goal()
        goal_space_shape = current_goal.shape
        #TODO: get the actual bonds here for proper normalization maybe?
        self.action_space = self.env.action_space
        self.env.task.task_params['time_threshold_in_goal_state_secs'] = self.env.dt
        if not is_goal_distance_dense:
            self.env.scale_reward_by_dt = False
        self.env.task.task_params['calculate_additional_dense_rewards'] = False
        self.env.task.set_sparse_reward(sparse_reward_weight)
        self.env.task.task_params['is_goal_distance_dense'] = \
            is_goal_distance_dense
        self.observation_space = spaces.Dict(dict(desired_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype=np.float32),
                                                  achieved_goal=spaces.Box(-np.inf,
                                                                          np.inf,
                                                                          shape=goal_space_shape,
                                                                          dtype=np.float32),
                                                  observation=self.env.observation_space))
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.env._add_wrapper_info({'her_environment': {'is_goal_distance_dense': is_goal_distance_dense,
                                                        'sparse_reward_weight': sparse_reward_weight}})

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
        obs_dict['achieved_goal'] = self.env.task.get_achieved_goal()
        obs_dict['desired_goal'] = self.env.task.get_desired_goal()
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        obs_dict = dict()
        normal_obs = self.env.reset(**kwargs)
        obs_dict['observation'] = normal_obs
        obs_dict['achieved_goal'] = self.env.task.get_achieved_goal()
        obs_dict['desired_goal'] = self.env.task.get_desired_goal()
        return obs_dict

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.task.compute_reward(achieved_goal,
                                            desired_goal,
                                            info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped
