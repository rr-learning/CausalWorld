from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion

import numpy as np
import math


def get_num_block_for_pyramid(pyramid_base_length):
    total_num_blocks = 0
    for i in range(pyramid_base_length):
        total_num_blocks += (pyramid_base_length - i)**2
    return total_num_blocks


class PyramidTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pyramid")
        self.task_params["pyramid_base_length"] = kwargs.get("pyramid_base_length", 4)
        self.task_params["randomize_pyramid_pose"] = kwargs.get("randomize_pyramid_pose", False)
        self.task_params["unit_length"] = kwargs.get("unit_length", 0.03)
        self.task_params["cube_color"] = kwargs.get("cube_color", np.array([1, 0, 0]))

        self.num_of_rigid_cubes = get_num_block_for_pyramid(self.task_params["pyramid_base_length"])
        self.pyramid_orientation = [0, 0, 0, 1]
        self.pyramid_position = None
        self.task_robot_observation_keys = ["joint_positions"]
        self.task_stage_observation_keys = []

    def _set_up_stage_arena(self):
        if not self.task_params["randomize_pyramid_pose"]:
            base_length = self.task_params["pyramid_base_length"]
            for i in range(base_length):
                self.stage.add_silhoutte_general_object(name="pyramid_level_{}".format(i),
                                                        shape="cube",
                                                        size=np.array([base_length - i, base_length - i, 1]) * self.task_params[
                                                            "unit_length"],
                                                        position=np.array([0, 0, 0.0115 + (2*i + 1) / 2 * self.task_params[
                                                            "unit_length"]]),
                                                        orientation=self.pyramid_orientation,
                                                        color=np.array([0, 1, 0]),
                                                        alpha=0.5)
        else:
            raise ValueError("randomize_pyramid_pose not supported yet")

        for i in range(self.num_of_rigid_cubes):
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.task_params["unit_length"] / 2)
            cube_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])

            self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                                shape="cube",
                                                size=np.array([1, 1, 1]) * self.task_params["unit_length"],
                                                position=cube_position,
                                                orientation=cube_orientation,
                                                color=self.task_params["cube_color"])

            self.task_stage_observation_keys.append("cube_{}_position".format(i))
            self.task_stage_observation_keys.append("cube_{}_orientation".format(i))
        return

    def _reset_task(self):
        sampled_positions = self.robot.sample_joint_positions()
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))
        for i in range(self.num_of_rigid_cubes):
            # TODO: For this we need more flexible sampling utils
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.task_params["unit_length"] / 2,
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = euler_to_quaternion([0, 0,
                                                    np.random.uniform(-np.pi, np.pi)])

            self.stage.set_objects_pose(names=["cube_{}".format(i)],
                                        positions=[cube_position],
                                        orientations=[cube_orientation])
        return

    def get_description(self):
        return "Task where the goal is to stack available cubes to form a pyramid"

    def get_reward(self):
        # TODO: placeholder 0 reward for now:
        reward = 0.0
        return reward

    def is_done(self):
        return self.task_solved