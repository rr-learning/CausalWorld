from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np
import copy
from itertools import compress
from causal_rl_bench.utils.state_utils import get_intersection


class StackedTowerImproperTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="stacked_tower_improper")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        # self.task_stage_observation_keys = ["block_position"]
        # self.task_params["block_mass"] = kwargs.get("block_mass", 0.02)
        self.task_params["randomize_joint_positions"] = kwargs.get(
            "randomize_joint_positions", True)
        self.task_params["levels_num"] = kwargs.get(
            "levels_num", 4)
        self.task_params["blocks_min_size"] = kwargs.get(
            "blocks_min_size", np.array([0.035, 0.035, 0.035]))
        self.task_params["block_mass"] = kwargs.get("block_mass", 0.08)
        self.current_stack_levels = self.task_params["levels_num"]
        self.intervention_spaces = None
        return

    def _generate_random_target(self, levels_num=4, min_size=np.array([0.035, 0.035, 0.035])):
        #generate first block
        limits = np.array(self.stage.floor_inner_bounding_box)
        start_z = self.stage.floor_height
        limits[0][0] = max(self.stage.floor_inner_bounding_box[0][0] + min_size[0],
                           limits[0][0])
        limits[1][0] = min(self.stage.floor_inner_bounding_box[1][0] - min_size[0],
                           limits[1][0])

        limits[0][1] = max(self.stage.floor_inner_bounding_box[0][1] + min_size[1],
                           limits[0][1])
        limits[1][1] = min(self.stage.floor_inner_bounding_box[1][1] - min_size[1],
                           limits[1][1])
        # first choose the center position
        position_x_y = np.random.uniform(limits[0][:2], limits[1][:2])
        # choose size width, depth, height
        allowed_max_width = min(self.stage.floor_inner_bounding_box[1][0] - position_x_y[0],
                                position_x_y[0] - self.stage.floor_inner_bounding_box[0][0]) * 2
        allowed_max_height = min(self.stage.floor_inner_bounding_box[1][1] - position_x_y[1],
                                 position_x_y[1] - self.stage.floor_inner_bounding_box[0][1]) * 2
        allowed_size = min(allowed_max_width, allowed_max_height)
        max_size = np.minimum([allowed_size, allowed_size, allowed_size], min_size*2)
        size = np.random.uniform(min_size, max_size)
        position_z = start_z + size[-1] / 2
        position = np.array([position_x_y[0], position_x_y[1], position_z])
        level_index = 0
        ranodm_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
        self.stage.add_silhoutte_general_object(name="target_"+ "level_"+str(level_index),
                                                shape="cube",
                                                position=position,
                                                orientation=ranodm_orientation,
                                                size=size)
        new_position = list(position)
        for level_index in range(1, levels_num):
            new_position[-1] = new_position[-1] + size[-1]
            ranodm_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
            self.stage.add_silhoutte_general_object(name="target_" + "level_" + str(level_index),
                                                    shape="cube",
                                                    position=new_position,
                                                    orientation=ranodm_orientation,
                                                    size=size)
        block_sizes = []
        for i in range(levels_num):
            block_sizes.append(size)
        return block_sizes

    def _set_up_stage_arena(self):
        self.current_stack_levels = self.task_params["levels_num"]
        block_sizes = self._generate_random_target(levels_num=self.current_stack_levels,
                                                   min_size=np.array([0.035, 0.035, 0.035]))
        for level_num in range(len(block_sizes)):
            self.stage.add_rigid_general_object(name="block_" + "level_" + str(level_num),
                                                shape="cube",
                                                size=block_sizes[level_num],
                                                colour=np.random.uniform(0, 1, size=[3]),
                                                mass=self.task_params["block_mass"])
            block_position = self.stage.random_position(height_limits=0.0425)
            block_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
            self.stage.set_objects_pose(names=["block_" + "level_" + str(level_num)],
                                        positions=[block_position],
                                        orientations=[block_orientation])
            trial_index = 0
            while not self.stage.check_feasiblity_of_stage() and trial_index < 10:
                block_position = self.stage.random_position(height_limits=[0.0425, 0.15])
                block_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
                self.stage.set_objects_pose(names=["block_" + "level_" + str(level_num)],
                                            positions=[block_position],
                                            orientations=[block_orientation])
                trial_index += 1
        self._set_intervention_spaces()
        return

    def _reset_task(self):
        #TODO: being able to rotate the whole shape/ target shape
        if self.task_params["randomize_joint_positions"]:
            #TODO: make sure of this method that it is a general one
            positions = self.robot.sample_joint_positions()
        else:
            positions = self.robot.get_rest_pose()[0]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))

        for rigid_object_name in self.stage.rigid_objects.keys():
            block_position = self.stage.random_position(height_limits=0.0425)
            block_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
            self.stage.set_objects_pose(names=[rigid_object_name],
                                        positions=[block_position],
                                        orientations=[block_orientation])
            trial_index = 0
            while not self.stage.check_feasiblity_of_stage() and trial_index < 10:
                block_position = self.stage.random_position(height_limits=[0.0425, 0.15])
                block_orientation = euler_to_quaternion([0, 0, np.random.uniform(-np.pi, np.pi)])
                self.stage.set_objects_pose(names=[rigid_object_name],
                                            positions=[block_position],
                                            orientations=[block_orientation])
                trial_index += 1

    def generate_new_goal(self):
        self.stage.remove_everything()
        self._set_up_stage_arena()
        self.reset_task()
        return

    def get_description(self):
        return "Task where the goal is to stack arbitrary shapes of cuboids"

    def get_reward(self):
        #intersection areas / union of all visual_objects
        intersection_area = 0
        union_area = 0 #TODO: under the assumption that the visual objects dont intersect
        for visual_object_key in self.stage.visual_objects:
            #TODO: distibuish between fixed and not
            visual_object = self.stage.get_object(visual_object_key)
            union_area += visual_object.get_area()
            for rigid_object_key in self.stage.rigid_objects:
                rigid_object = self.stage.get_object(rigid_object_key)
                if rigid_object.is_not_fixed:
                    intersection_area += get_intersection(visual_object.get_bounding_box(),
                                                          rigid_object.get_bounding_box())
        # #TODO: discuss termination conditions
        # # if abs(z - target_height) < 0.02:
        # #     self.task_solved = True
        return intersection_area / float(union_area)

    # def do_random_intervention(self):
    #     #choose random variable one intervention  only and intervene
    #     variable_name = np.random.choice(list(self.intervention_spaces.keys()))
    #     variable_space = self.intervention_spaces[variable_name]
    #     sub_variable_name = None
    #     #if its a block then choose a property
    #     if isinstance(variable_space, dict):
    #         sub_variable_name = np.random.choice(list(variable_space.keys()))
    #         variable_space = variable_space[sub_variable_name]
    #     self.do_intervention(variable_name, np.random.uniform(variable_space[0], variable_space[1]),
    #                          sub_variable_name=sub_variable_name)
    #     return
    #
    # def do_intervention(self, variable_name, variable_value, sub_variable_name=None):
    #     if sub_variable_name is not None:
    #         if sub_variable_name == "orientation":
    #             variable_value = euler_to_quaternion(variable_value)
    #         interventions_dict = dict()
    #         interventions_dict[sub_variable_name] = variable_value
    #         self.stage.object_intervention(variable_name, interventions_dict)
    #     elif variable_name == 'stack_colour':
    #         interventions_dict = dict()
    #         interventions_dict["colour"] = variable_value
    #         for visual_object_name in self.stage.visual_objects.keys():
    #             self.stage.object_intervention(visual_object_name, interventions_dict)
    #     elif variable_name == 'stack_levels':
    #         #remove levels now
    #         if self.current_stack_levels > variable_value:
    #             for i in range(variable_value, self.current_stack_levels):
    #                 self.stage.remove_general_object("target_"+"level_" + str(i))
    #                 select_rigid_objects_to_remove = [key.startswith("block_" + "level_" + str(i))
    #                                                   for key in self.stage.rigid_objects]
    #                 rigid_objects_to_remove = list(compress(list(self.stage.rigid_objects.keys()),
    #                                                         select_rigid_objects_to_remove))
    #                 for rigid_object_to_remove in rigid_objects_to_remove:
    #                     self.stage.remove_general_object(rigid_object_to_remove)
    #             self.current_stack_levels = variable_value
    #             self._set_intervention_spaces()
    #         #TODO: add levels support
    #     return

    def _set_intervention_spaces(self):
        self.intervention_spaces = dict()
        for name in self.stage.rigid_objects.keys():
            self.intervention_spaces[name] = dict()
            self.intervention_spaces[name]['position'] = np.array([[-0.15, -0.15, -0.15], [0.15, 0.15, 0.15]])
            self.intervention_spaces[name]['orientation'] = np.array([[0, 0, 0], [np.pi, np.pi, np.pi]])
            self.intervention_spaces[name]['colour'] = np.array([[0, 0, 0], [1, 1, 1]])
            self.intervention_spaces[name]['mass'] = np.array([0.02, 0.1])
        self.intervention_spaces['stack_levels'] = np.array([1, 8])
        self.intervention_spaces['stack_colour'] = np.array([[0, 0, 0], [1, 1, 1]])

    def get_intervention_spaces(self):
        return self.intervention_spaces



