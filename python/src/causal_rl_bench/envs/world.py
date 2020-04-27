import numpy as np
import time
import pybullet
import gym

from pybullet_fingers.sim_finger import SimFinger
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.tasks.task import Task


class World(gym.Env):
    """
    Base environment of the robot manipulation task
    """

    def __init__(
        self, task_id="pushing", control_rate=0.003, enable_visualization=True, seed=0,
            action_mode="joint_position",
    ):
        """
        Constructor sets up the physical world parameters,
        and resets to begin training.

        Args:
            control_rate_s (float): the rate at which the env step runs
            finger-type (str "single"/"tri"): to train on the "single"
                or the "tri" finger
            enable_visualization (bool): if the simulation env is to be
                visualized
        """

        gym.Env.__init__(self)

        self.robot = TriFingerRobot()
        self.stage = Stage()
        self.tasks = []

    def step(self, action):
        raise Exception(" ")

    def sample_new_task(self):
        raise Exception(" ")

    def switch_task(self):
        raise Exception(" ")

    def get_counterfactual_world(self):
        raise Exception(" ")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        raise Exception(" ")

    def close(self):
        raise Exception(" ")

    def render(self, mode='human'):
        raise Exception(" ")

    def enforce_max_episode_length(self, episode_length=2000):
        raise Exception(" ")

    def _is_done(self):
        return self.tasks.is_done()

    def get_full_state(self):
        raise Exception(" ")

    def set_full_state(self):
        raise Exception(" ")
