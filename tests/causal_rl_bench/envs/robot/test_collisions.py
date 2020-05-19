from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
from causal_rl_bench.envs.world import World
import numpy as np
import unittest


class CrossReferenceWPybulletFingers(unittest.TestCase):
    def setUp(self):
        task = Task(task_id='pushing')
        self.causal_rl_env = World(task=task,
                                   enable_visualization=True,
                                   seed=0,
                                   skip_frame=1,
                                   normalize_actions=True,
                                   normalize_observations=True)
        return

    def test_1(self):
        """Will succeed"""
        for _ in range(5):
            for _ in range(1000):
                action = self.causal_rl_env.action_space.sample()
                obs2, reward2, done, info = self.causal_rl_env.step(action)
                if self.causal_rl_env.robot.is_colliding_with_stage():
                    print("hello")

    def tearDown(self):
        self.causal_rl_env.close()


if __name__ == "__main__":
    unittest.main()
