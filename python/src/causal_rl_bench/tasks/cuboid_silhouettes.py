from counterfactual.python.src.causal_rl_bench.tasks.task import Task
from scipy.spatial.transform import Rotation as rotation
import numpy as np
import math


class CuboidSilhouette(Task):
    def __init__(self, silhouette_size=None, silhouette_position_mode="center", unit_length=0.065, cube_color="red"):
        super().__init__()
        self.id = "cuboid_silhouette"
        self.robot = None
        self.stage = None
        if silhouette_size is None:
            self.silhouette_size = np.array([1, 2, 3])
        else:
            self.silhouette_size = silhouette_size
        self.num_of_rigid_cubes = int(np.prod(self.silhouette_size))
        self.silhouette_position_mode = silhouette_position_mode
        self.silhouette_orientation = [0, 0, 0, 1]
        self.silhouette_subgoals = []
        self.unit_length = unit_length
        self.cube_color = cube_color

        self.task_solved = False
        self.observation_keys = []

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage

        if self.robot.get_observation_mode() == "structured":
            self.observation_keys = ["joint_positions",
                                     "silhouette_cuboid_target_position",
                                     "silhouette_cuboid_target_orientation",
                                     "silhouette_cuboid_target_size"]
        elif self.robot.get_observation_mode() == "cameras":
            self.observation_keys = ["cameras"]
        else:
            raise ValueError("Observation mode not supported for this task")

        if self.silhouette_position_mode == "center":
            self.silhouette_position = np.array([0, 0, 0.0115 + self.silhouette_size[2] / 2 * self.unit_length])
        else:
            raise ValueError("please provide valid silhouette position argument")

        self.stage.add_silhoutte_general_object(name="cuboid_target",
                                                shape="cube",
                                                size=self.silhouette_size * self.unit_length,
                                                position=self.silhouette_position,
                                                orientation=self.silhouette_orientation,
                                                colour="green")

        for i in range(self.num_of_rigid_cubes):
            # TODO: For this we need more flexible sampling utils
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.unit_length / 2,
                                                       radius_limits=(0.05, 0.13),
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = rotation.from_euler('z', np.random.uniform(0, 360),
                                                   degrees=True).as_quat()
            self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                                shape="cube",
                                                size=np.array([1, 1, 1]) * self.unit_length,
                                                position=cube_position,
                                                orientation=cube_orientation,
                                                colour=self.cube_color)

            if self.robot.get_observation_mode() == "structured":
                self.observation_keys.append("rigid_cube_{}_position".format(i))
                self.observation_keys.append("rigid_cube_{}_orientation".format(i))

        self.stage.finalize_stage()

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(sampled_positions)

        self.task_solved = False
        self.reset_scene_objects()

        return self.robot.get_current_full_observations()

    def get_description(self):
        return "Task where the goal is to stack available cubes into a target silhouette"

    def get_reward(self):
        reward = 0.0
        return reward

    def is_terminated(self):
        return self.task_solved

    def filter_observations(self, robot_observations_dict,
                            stage_observations_dict):
        full_observations_dict = dict(robot_observations_dict)
        full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.observation_keys:
            observations_filtered = \
                np.append(observations_filtered,
                          np.array(full_observations_dict[key]))
        return observations_filtered

    def get_counterfactual_variant(self, **kwargs):
        # TODO: This is an example for counterfactual worlds where color and/or scale changes
        if "cube_color" in kwargs.keys():
            self.cube_color = kwargs["cube_color"]
        elif "unit_length" in kwargs.keys():
            self.unit_length = kwargs["unit_length"]
        else:
            raise ValueError("Not supported variable for counterfactual reasoning")
        return CuboidSilhouette(silhouette_size=self.silhouette_size,
                                silhouette_position_mode=self.silhouette_position_mode,
                                unit_length=self.unit_length,
                                cube_color=self.cube_color)


    def reset_scene_objects(self):
        for i in range(self.num_of_rigid_cubes):
            # TODO: For this we need more flexible sampling utils
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.unit_length / 2,
                                                       radius_limits=(0.05, 0.13),
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = rotation.from_euler('z', np.random.uniform(0, 360),
                                                   degrees=True).as_quat()

            self.stage.set_states(names=["cube_{}".format(i)],
                                  positions=[cube_position],
                                  orientations=[cube_orientation])
