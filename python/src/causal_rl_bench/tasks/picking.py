from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np


class PickingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="picking")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions"]
        self.task_stage_observation_keys = ["block_position"]
        self.task_params["block_mass"] = kwargs.get("block_mass", 0.02)
        self.task_params["randomize_joint_positions"] = kwargs.get(
            "randomize_joint_positions", True)
        self.task_params["randomize_block_pose"] = kwargs.get(
            "randomize_block_pose", True)
        self.task_params["goal_height"] = kwargs.get("goal_height", 0.1)

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube",
                                            mass=self.task_params["block_mass"])
        return

    def _reset_task(self):
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_positions()
        else:
            positions = [0, -0.5, -0.6,
                         0, -0.4, -0.7,
                         0, -0.4, -0.7]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))

        # reset stage next
        if self.task_params["randomize_block_pose"]:
            block_position = self.stage.random_position(height_limits=0.0425)
            block_orientation = euler_to_quaternion([0, 0,
                                                     np.random.uniform(-np.pi,
                                                                       np.pi)])
        else:
            block_position = [0.0, -0.02, 0.045155]
            block_orientation = euler_to_quaternion([0, 0, 0.0])
        self.stage.set_objects_pose(names=["block"],
                                    positions=[block_position],
                                    orientations=[block_orientation])
        return

    def get_description(self):
        return "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_position = self.stage.get_object_state('block', 'position')
        target_height = self.task_params["goal_height"]
        x = block_position[0]
        y = block_position[1]
        z = block_position[2]
        reward = -abs(z - target_height) - (x**2 + y**2)
        if abs(z - target_height) < 0.02:
            self.task_solved = True
        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        new_size = np.random.uniform([0.065], [0.15], size=[3, ])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("block", interventions_dict)
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

