from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
from causal_rl_bench.envs.world import World
from causal_rl_bench.utils.state_utils import get_iou
import numpy as np
import unittest


class CrossReferenceWPybulletFingers(unittest.TestCase):
    def setUp(self):
        task = Task(task_generator_id='picking', randomize_joint_positions=False,
                    randomize_block_pose=False, block_mass=0.02,
                    goal_height=0.0425)
        self.causal_rl_env = World(task=task,
                                   enable_visualization=True,
                                   seed=0,
                                   skip_frame=1,
                                   normalize_actions=True,
                                   normalize_observations=True)
        return

    def test_iou_calculation(self):
        """Will succeed"""
        for _ in range(5):
            self.causal_rl_env.reset()
            for _ in range(5):
                action = self.causal_rl_env.action_space.sample()
                obs2, reward2, done, info = self.causal_rl_env.step(action)
                assert(get_iou(self.causal_rl_env.__stage.get_object('goal_position').get_bounding_box(),
                               self.causal_rl_env.__stage.get_object('block').get_bounding_box(),
                               self.causal_rl_env.__stage.get_object('goal_position').get_volume(),
                               self.causal_rl_env.__stage.get_object('goal_position').get_volume()) > 0.9)

    def tearDown(self):
        self.causal_rl_env.close()


if __name__ == "__main__":
    unittest.main()
