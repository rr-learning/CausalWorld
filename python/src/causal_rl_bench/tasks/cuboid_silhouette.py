from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np
import math
import itertools


class CuboidSilhouette(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="cuboid_silhouette")
        self.task_params["silhouette_size"] = kwargs.get("silhouette_size", np.array([1, 2, 3]))
        self.task_params["silhouette_position_mode"] = kwargs.get("silhouette_position_mode", "center")
        self.task_params["unit_length"] = kwargs.get("unit_length", 0.065)
        self.task_params["cube_color"] = kwargs.get("cube_color", "FF0000")
        self.num_of_rigid_cubes = int(np.prod(self.task_params["silhouette_size"]))
        self.silhouette_orientation = [0, 0, 0, 1]
        self.silhouette_position = None
        self.s_xhigher = None
        self.s_xlower = None
        self.s_yhigher = None
        self.s_ylower = None
        self.s_zhigher = None
        self.s_zlower = None
        self.task_robot_observation_keys = ["joint_positions"]
        self.task_stage_observation_keys = ["cuboid_target_position",
                                            "cuboid_target_orientation",
                                            "cuboid_target_size"]

    def _set_up_stage_arena(self):
        if self.task_params["silhouette_position_mode"] == "center":
            self.silhouette_position = np.array([0, 0, 0.0115 +
                                                 self.task_params["silhouette_size"][2] / 2 * self.task_params["unit_length"]])
            # TODO: this needs to be implemented for a general silhouette orientation and position
            self.s_xhigher = self.task_params["silhouette_size"][0] / 2 * self.task_params["unit_length"]
            self.s_xlower = - self.task_params["silhouette_size"][0] / 2 * self.task_params["unit_length"]
            self.s_yhigher = self.task_params["silhouette_size"][1] / 2 * self.task_params["unit_length"]
            self.s_ylower = - self.task_params["silhouette_size"][1] / 2 * self.task_params["unit_length"]
            self.s_zhigher = 0.0115 + self.task_params["silhouette_size"][2] * self.task_params["unit_length"]
            self.s_zlower = 0.0115
        else:
            raise ValueError("please provide valid silhouette position argument")

        self.stage.add_silhoutte_general_object(name="cuboid_target",
                                                shape="cube",
                                                size=self.task_params["silhouette_size"] * self.task_params["unit_length"],
                                                position=self.silhouette_position,
                                                orientation=self.silhouette_orientation,
                                                colour=np.array([0, 1, 0]),
                                                alpha=0.5)

        for i in range(self.num_of_rigid_cubes):
            # TODO: For this we need more flexible sampling utils
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.task_params["unit_length"] / 2,
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = euler_to_quaternion([0, 0,
                                                    np.random.uniform(-np.pi, np.pi)])
            r, g, b = bytes.fromhex(self.task_params["cube_color"])
            self.stage.add_rigid_general_object(name="cube_{}".format(i),
                                                shape="cube",
                                                size=np.array([1, 1, 1]) * self.task_params["unit_length"],
                                                position=cube_position,
                                                orientation=cube_orientation,
                                                colour=np.array([r, g, b]))

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
        return "Task where the goal is to stack available cubes into a target silhouette"

    def get_reward(self):
        reward = 0.0
        for i in range(self.num_of_rigid_cubes):
            cuboid_position = self.stage.get_object_state("cube_{}".format(i), 'position')
            cuboid_orientation = self.stage.get_object_state("cube_{}".format(i), 'orientation')
            reward += self.reward_per_cuboid(cuboid_position=cuboid_position,
                                             cuboid_orientation=cuboid_orientation,
                                             cuboid_size=np.array([1, 1, 1]) * self.task_params["unit_length"])
        return reward


    #TODO: discuss whats the difference between this and do random intervention
    def get_counterfactual_variant(self, **kwargs):
        # TODO: This is an example for counterfactual worlds where color and/or scale changes
        if "cube_color" in kwargs.keys():
            self.task_params["cube_color"] = kwargs["cube_color"]
        elif "unit_length" in kwargs.keys():
            self.task_params["unit_length"] = kwargs["unit_length"]
        else:
            raise ValueError("Not supported variable for counterfactual reasoning")
        return CuboidSilhouette(silhouette_size=self.task_params["silhouette_size"],
                                silhouette_position_mode=self.task_params["silhouette_position_mode"],
                                unit_length=self.task_params["unit_length"],
                                cube_color=self.task_params["cube_color"])

    def reward_per_cuboid(self, cuboid_position, cuboid_orientation, cuboid_size):
        cuboid_reward = 0
        for unit_vertex_tuple in itertools.product([-1, 1], repeat=3):
            vertex_position_in_cube_frame = np.multiply(np.array(unit_vertex_tuple), cuboid_size / 2)
            #TODO: avoid using pybullet directly get the pybullet client from robot or stage
            vertex_coords, _ = self.robot.get_pybullet_client().multiplyTransforms(positionA=cuboid_position,
                                                                                   orientationA=[0, 0, 0, 1],
                                                                                   positionB=vertex_position_in_cube_frame,
                                                                                   orientationB=cuboid_orientation)
            vertex_distance = 0
            if vertex_coords[0] > self.s_xhigher:
                vertex_distance += vertex_coords[0] - self.s_xhigher
            elif vertex_coords[0] < self.s_xlower:
                vertex_distance += self.s_xlower - vertex_coords[0]

            if vertex_coords[1] > self.s_yhigher:
                vertex_distance += vertex_coords[1] - self.s_yhigher
            elif vertex_coords[1] < self.s_ylower:
                vertex_distance += self.s_ylower - vertex_coords[1]

            if vertex_coords[2] > self.s_zhigher:
                vertex_distance += vertex_coords[2] - self.s_zhigher
            elif vertex_coords[2] < self.s_zlower:
                vertex_distance += self.s_zlower - vertex_coords[2]
            cuboid_reward -= vertex_distance
        return cuboid_reward

    def do_random_intervention(self):
        interventions_dict = dict()
        self.task_params["unit_length"] = np.random.uniform(low=0.04, high=0.08)

        for i in range(self.num_of_rigid_cubes):
            min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
            max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
            cube_position = self.stage.random_position(height_limits=0.0115 + self.task_params["unit_length"] / 2,
                                                       angle_limits=(min_angle, max_angle))
            cube_orientation = euler_to_quaternion([0, 0,
                                                    np.random.uniform(-np.pi, np.pi)])

            new_colour = np.random.uniform([0], [1], size=[3, ])
            interventions_dict["position"] = cube_position
            interventions_dict["orientation"] = cube_orientation
            interventions_dict["colour"] = new_colour
            interventions_dict["size"] = np.array([1, 1, 1]) * self.task_params["unit_length"]
            self.stage.object_intervention("cube_{}".format(i), interventions_dict)

        # For target silhouette

        self.silhouette_position = np.array([0, 0, 0.0115 + self.task_params["silhouette_size"][2] / 2 * self.task_params["unit_length"]])
        # TODO: this needs to be implemented for a general silhouette orientation
        self.s_xhigher = self.task_params["silhouette_size"][0] / 2 * self.task_params["unit_length"]
        self.s_xlower = - self.task_params["silhouette_size"][0] / 2 * self.task_params["unit_length"]
        self.s_yhigher = self.task_params["silhouette_size"][1] / 2 * self.task_params["unit_length"]
        self.s_ylower = - self.task_params["silhouette_size"][1] / 2 * self.task_params["unit_length"]
        self.s_zhigher = 0.0115 + self.task_params["silhouette_size"][2] / 2 * self.task_params["unit_length"]
        self.s_zlower = 0.0115

        interventions_dict_target = dict()
        interventions_dict_target["position"] = self.silhouette_position
        interventions_dict_target["orientation"] = [0, 0, 0, 1]
        interventions_dict_target["colour"] = np.random.uniform([0], [1], size=[3, ])
        interventions_dict_target["size"] = self.task_params["silhouette_size"] * self.task_params["unit_length"]
        self.stage.object_intervention("cuboid_target", interventions_dict_target)

    def do_intervention(self, **kwargs):
        # TODO: For now we only support color for testing reasons
        if "cube_color" not in kwargs.keys():
            raise Exception("Only intervention on cube color allowed at the moment")

        if "cube_color" in kwargs.keys():
            interventions_dict = dict()
            self.task_params["cube_color"] = kwargs.get("cube_color")
            r, g, b = bytes.fromhex(self.task_params["cube_color"])
            for i in range(self.num_of_rigid_cubes):
                min_angle = i / self.num_of_rigid_cubes * 2 * math.pi
                max_angle = (i + 1) / self.num_of_rigid_cubes * 2 * math.pi
                cube_position = self.stage.random_position(height_limits=0.0115 + self.task_params["unit_length"] / 2,
                                                           angle_limits=(min_angle, max_angle))
                cube_orientation = euler_to_quaternion([0, 0,
                                                        np.random.uniform(-np.pi, np.pi)])
                interventions_dict["position"] = cube_position
                interventions_dict["orientation"] = cube_orientation
                interventions_dict["colour"] = np.array([r, g, b])
                interventions_dict["size"] = np.array([1, 1, 1]) * self.task_params["unit_length"]
                self.stage.object_intervention("cube_{}".format(i), interventions_dict)

    def get_visual_variables(self):
        variables = {"cube_color": ["FF0000",
                                    "00FF00",
                                    "0000FF"]}
        return variables
