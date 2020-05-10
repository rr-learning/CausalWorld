from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
from causal_rl_bench.envs.world import World
import numpy as np
import unittest


class PyBulletFingersPickingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pybullet_fingers"
                                   "_cross_reference_picking")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions"]
        self.task_stage_observation_keys = ["block_position"]
        self.task_params["block_mass"] = kwargs.get("block_mass", 0.08)

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube",
                                            mass=self.task_params[
                                                "block_mass"],
                                            position=np.array(
                                                [0.15, 0.0, 0.0425]),
                                            orientation=np.array([0, 0, 0, 1]))
        return

    def _reset_task(self):
        # reset stage next
        block_position = self.stage.legacy_random_position(height_limits=0.0425)
        block_orientation = euler_to_quaternion([0, 0, 0])
        self.stage.set_objects_pose(names=["block"],
                                    positions=[block_position],
                                    orientations=[block_orientation])
        positions = self.robot.sample_positions(
            sampling_strategy="separated")
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))
        return

    def get_description(self):
        return "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_position = self.stage.get_object_state('block', 'position')
        end_effector_positions = self.robot.compute_end_effector_positions(
            self.robot.latest_full_state)
        end_effector_positions = end_effector_positions.reshape(-1, 3)
        distance_from_block = np.sum((end_effector_positions - block_position)
                                     ** 2)
        z = block_position[2]
        reward = z - distance_from_block
        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")


class CrossReferenceWPybulletFingers(unittest.TestCase):
    def setUp(self):
        import gym
        self.pybullet_env = gym.make(
            "pybullet_fingers.gym_wrapper:pick-v0",
            control_rate_s=0.004,
            enable_visualization=False,
            seed=0
        )
        task = PyBulletFingersPickingTask()
        self.causal_rl_env = World(task=task,
                                   enable_visualization=False,
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
                obs1, reward1, done, info = self.pybullet_env.step(action)
                obs2, reward2, done, info = self.causal_rl_env.step(action)
                assert(np.array_equal(obs1, obs2))
                assert (np.array_equal(reward1, reward2))

    def tearDown(self):
        self.causal_rl_env.close()
        self.pybullet_env.close()


if __name__ == "__main__":
    unittest.main()
