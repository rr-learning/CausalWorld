from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion

import numpy as np
import math


class ArchTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="arch")
        self.task_params["cube_color"] = kwargs.get("cube_color", np.array([1, 0, 0]))

        self.num_of_rigid_cubes = 4
        self.arch_orientation = [0, 0, 0, 1]
        self.arch_position = None
        self.task_robot_observation_keys = ["joint_positions"]
        self.task_stage_observation_keys = []

    def _set_up_stage_arena(self):

        self.stage.add_silhoutte_general_object(name="pillar_1",
                                                shape="cube",
                                                size=np.array([0.04, 0.1, 0.15]),
                                                position=np.array([-0.05, 0, 0.0115 + 0.075]),
                                                orientation=self.arch_orientation,
                                                colour=np.array([0, 1, 0]),
                                                alpha=0.5)

        self.stage.add_silhoutte_general_object(name="pillar_2",
                                                shape="cube",
                                                size=np.array([0.04, 0.08, 0.15]),
                                                position=np.array([0.05, 0, 0.0115 + 0.075]),
                                                orientation=self.arch_orientation,
                                                colour=np.array([0, 1, 0]),
                                                alpha=0.5)

        self.stage.add_silhoutte_general_object(name="bar",
                                                shape="cube",
                                                size=np.array([0.14, 0.06, 0.04]),
                                                position=np.array([0, 0, 0.0115 + 0.15 + 0.02]),
                                                orientation=self.arch_orientation,
                                                colour=np.array([0, 1, 0]),
                                                alpha=0.5)

        self.stage.add_silhoutte_general_object(name="cargo",
                                                shape="cube",
                                                size=np.array([0.04, 0.04, 0.04]),
                                                position=np.array([0, 0, 0.0115 + 0.15 + 0.04 + 0.02]),
                                                orientation=self.arch_orientation,
                                                colour=np.array([0, 1, 0]),
                                                alpha=0.5)

        i = 0
        cube_position = self.stage.random_position(height_limits=0.0115 + 0.04 / 2)
        cube_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                            shape="cube",
                                            size=np.array([0.1, 0.15, 0.04]),
                                            position=cube_position,
                                            orientation=cube_orientation,
                                            colour=self.task_params["cube_color"])

        self.task_stage_observation_keys.append("cube_{}_position".format(i))
        self.task_stage_observation_keys.append("cube_{}_orientation".format(i))

        i = 1
        cube_position = self.stage.random_position(height_limits=0.0115 + 0.04 / 2)
        cube_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                            shape="cube",
                                            size=np.array([0.15, 0.08, 0.04]),
                                            position=cube_position,
                                            orientation=cube_orientation,
                                            colour=self.task_params["cube_color"])

        self.task_stage_observation_keys.append("cube_{}_position".format(i))
        self.task_stage_observation_keys.append("cube_{}_orientation".format(i))

        i = 2
        cube_position = self.stage.random_position(height_limits=0.0115 + 0.04 / 2)
        cube_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                            shape="cube",
                                            size=np.array([0.14, 0.06, 0.04]),
                                            position=cube_position,
                                            orientation=cube_orientation,
                                            colour=self.task_params["cube_color"])

        self.task_stage_observation_keys.append("cube_{}_position".format(i))
        self.task_stage_observation_keys.append("cube_{}_orientation".format(i))

        i = 3
        cube_position = self.stage.random_position(height_limits=0.0115 + 0.04 / 2)
        cube_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                            shape="cube",
                                            size=np.array([0.04, 0.04, 0.04]),
                                            position=cube_position,
                                            orientation=cube_orientation,
                                            colour=self.task_params["cube_color"])

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
            cube_position = self.stage.random_position(height_limits=0.0115 + 0.04 / 2,
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = euler_to_quaternion([0, 0,
                                                    np.random.uniform(-np.pi, np.pi)])

            self.stage.set_objects_pose(names=["cube_{}".format(i)],
                                        positions=[cube_position],
                                        orientations=[cube_orientation])
        return

    def get_description(self):
        return "Task where the goal is to stack available cubes to form a predefined arch"

    def get_reward(self):
        # TODO: placeholder 0 reward for now:
        reward = 0.0
        return reward

    def is_done(self):
        return self.task_solved