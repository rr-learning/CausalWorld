from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np
import copy


class TowersGeneratorTask(BaseTask):
    def __init__(self, **kwargs):
        """
        This task generator
        will most probably deal with camera data if u want to use the
        sample goal function
        :param kwargs:
        """
        super().__init__(task_name="towers",
                         use_train_space_only=kwargs.get("use_train_space_only",
                                                         False),
                         fractional_reward_weight=
                         kwargs.get("fractional_reward_weight", 0),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([750,
                                              125,
                                              250,
                                              0.005])))
        self._task_robot_observation_keys = ["time_left_for_task",
                                             "joint_positions",
                                             "joint_velocities",
                                             "end_effector_positions"]

        # for this task the stage observation keys will be set with the
        # goal/structure building
        self._task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.08)
        self._task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self._task_params["number_of_blocks_in_tower"] = \
            kwargs.get("number_of_blocks_in_tower", np.array([1, 1, 2]))
        self._task_params["tower_dims"] = \
            kwargs.get("tower_dims", np.array([0.065, 0.065, 0.13]))
        self._task_params["tower_center"] = \
            kwargs.get("tower_center", np.array([0, 0]))
        self.current_tower_dims = np.array(self._task_params["tower_dims"])
        self.current_number_of_blocks_in_tower = \
            np.array(self._task_params["number_of_blocks_in_tower"])
        self.current_tool_block_mass = float(self._task_params["tool_block_mass"])
        self.current_tower_center = np.array(self._task_params["tower_center"])

    def get_description(self):
        """

        :return:
        """
        return "Task where the goal is to stack arbitrary number of towers side by side"

    # TODO: add obstacles interventions? up to a 10 obstacles?
    def _set_up_stage_arena(self):
        """

        :return:
        """
        self._set_up_cuboid(self.current_tower_dims,
                            self.current_number_of_blocks_in_tower,
                            self.current_tower_center)
        return

    def _set_up_cuboid(self, tower_dims, number_of_blocks_in_tower,
                       center_position):
        """

        :param tower_dims:
        :param number_of_blocks_in_tower:
        :param center_position:
        :return:
        """
        self._stage.remove_everything()
        joint_positions = self._robot.get_upper_joint_positions()
        self._robot.set_full_state(np.append(joint_positions,
                                             np.zeros(9)))
        self._task_stage_observation_keys = []
        block_size = tower_dims / number_of_blocks_in_tower
        curr_height = 0 - block_size[-1] / 2
        rigid_block_position = np.array([-0.12, -0.12, 0 + block_size[-1] / 2])
        silhouettes_creation_dicts = []
        for level in range(number_of_blocks_in_tower[-1]):
            curr_height += block_size[-1]
            curr_y = center_position[1] - tower_dims[1] / 2 - block_size[1] / 2
            for col in range(number_of_blocks_in_tower[1]):
                curr_y += block_size[1]
                curr_x = center_position[0] - tower_dims[0] / 2 - block_size[0] / 2
                for row in range(number_of_blocks_in_tower[0]):
                    curr_x += block_size[0]
                    creation_dict = {'name': "tool_" + "level_" +
                                             str(level) + "_col_" +
                                             str(col) + "_row_" + str(row),
                                     'shape': "cube",
                                     'initial_position': np.copy(rigid_block_position),
                                     'initial_orientation': [0, 0, 0, 1],
                                     'mass': self.current_tool_block_mass,
                                     'size': block_size}
                    self._stage.add_rigid_general_object(**creation_dict)
                    creation_dict = {'name': "goal_" + "level_" +
                                             str(level) + "_col_" +
                                             str(col) + "_row_" + str(row),
                                     'shape': "cube",
                                     'position': [curr_x, curr_y, curr_height],
                                     'orientation': [0, 0, 0, 1],
                                     'size': block_size}
                    silhouettes_creation_dicts.append(copy.deepcopy(creation_dict))
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_type')
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_size')
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_cartesian_position')
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_orientation')
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_linear_velocity')
                    self._task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_angular_velocity')
                    self._task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_type')
                    self._task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_size')
                    self._task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_cartesian_position')
                    self._task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_orientation')
                    rigid_block_position[:2] += block_size[:2]
                    rigid_block_position[:2] += 0.005
                    if np.any(rigid_block_position[:2] > np.array([0.12, 0.12])):
                        rigid_block_position[0] = -0.12
                        rigid_block_position[1] = -0.12
                        rigid_block_position[2] = rigid_block_position[2] + block_size[-1] / 2

        for i in range(len(silhouettes_creation_dicts)):
            self._stage.add_silhoutte_general_object(**silhouettes_creation_dicts[i])
        return

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
        # for now remove all possible interventions on the goal in general
        # intevrntions on size of objects might become tricky to handle
        # contradicting interventions here?
        super(TowersGeneratorTask, self)._set_training_intervention_spaces()
        for visual_object in self._stage.get_visual_objects():
            del self._training_intervention_spaces[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._training_intervention_spaces[rigid_object]['size']
        self._training_intervention_spaces['number_of_blocks_in_tower'] = \
            np.array([[1, 1, 1], [4, 4, 4]])
        self._training_intervention_spaces['blocks_mass'] = \
            np.array([0.02, 0.06])
        self._training_intervention_spaces['tower_dims'] = \
            np.array([[0.035, 0.035, 0.035], [0.10, 0.10, 0.10]])
        self._training_intervention_spaces['tower_center'] = \
            np.array([[-0.1, -0.1], [0.05, 0.05]])
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        super(TowersGeneratorTask, self)._set_testing_intervention_spaces()
        for visual_object in self._stage.get_visual_objects():
            del self._testing_intervention_spaces[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._testing_intervention_spaces[rigid_object]['size']
        self._testing_intervention_spaces['number_of_blocks_in_tower'] = \
            np.array([[4, 4, 4], [6, 6, ]])
        self._testing_intervention_spaces['blocks_mass'] = \
            np.array([0.06, 0.08])
        self._testing_intervention_spaces['tower_dims'] = \
            np.array([[0.10, 0.10, 0.10], [0.13, 0.13, 0.13]])
        self._testing_intervention_spaces['tower_center'] = \
            np.array([[0.05, 0.05], [0.1, 0.1]])
        return

    def sample_new_goal(self, training=True, level=None):
        """

        :param training:
        :param level:
        :return:
        """
        intervention_dict = dict()
        if training:
            intervention_space = self._training_intervention_spaces
        else:
            intervention_space = self._testing_intervention_spaces
        intervention_dict['number_of_blocks_in_tower'] = [np. \
            random.randint(intervention_space['number_of_blocks_in_tower'][0][i],
                           intervention_space['number_of_blocks_in_tower'][1][i]) for i in range(3)]
        intervention_dict['blocks_mass'] = np. \
            random.uniform(intervention_space['blocks_mass'][0],
                           intervention_space['blocks_mass'][1])
        intervention_dict['tower_dims'] = np. \
            random.uniform(intervention_space['tower_dims'][0],
                           intervention_space['tower_dims'][1])
        intervention_dict['tower_center'] = np. \
            random.uniform(intervention_space['tower_center'][0],
                           intervention_space['tower_center'][1])
        return intervention_dict

    def get_task_generator_variables_values(self):
        """

        :return:
        """
        return {'tower_dims': self.current_tower_dims,
                'blocks_mass': self.current_tool_block_mass,
                'number_of_blocks_in_tower': self.current_number_of_blocks_in_tower,
                'tower_center': self.current_tower_center}

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        # TODO: support level removal intervention
        if len(interventions_dict) == 0:
            return True, False
        reset_observation_space = True
        if "tower_dims" in interventions_dict:
            self.current_tower_dims = interventions_dict["tower_dims"]
        if "number_of_blocks_in_tower" in interventions_dict:
            self.current_number_of_blocks_in_tower = interventions_dict["number_of_blocks_in_tower"]
        if "blocks_mass" in interventions_dict:
            self.current_tool_block_mass = interventions_dict["blocks_mass"]
        if "tower_center" in interventions_dict:
            self.current_tower_center = interventions_dict["tower_center"]
        # TODO: tae care of center and orientation seperatly since we dont need to recreate everything,
        # just translate and rotate!!
        if "tower_dims" in interventions_dict or "number_of_blocks_in_tower" in interventions_dict or \
                "tower_center" in interventions_dict or "tower_orientation" in interventions_dict:
            self._set_up_cuboid(tower_dims=self.current_tower_dims,
                                number_of_blocks_in_tower=self.current_number_of_blocks_in_tower,
                                center_position=self.current_tower_center)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self._stage.get_rigid_objects():
                if self._stage.get_rigid_objects()[rigid_object].is_not_fixed:
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.current_tool_block_mass
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_testing_intervention_spaces()
        self._set_training_intervention_spaces()
        self._stage.finalize_stage()
        return True, reset_observation_space

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:
        :return:
        """
        end_effector_positions = self._robot.get_latest_full_state()['end_effector_positions']
        end_effector_positions = end_effector_positions.reshape(-1, 3)
        joint_velocities = self._robot.get_latest_full_state()['velocities']

        block_level_0_position = self._stage.get_object_state('tool_level_0_col_0_row_0',
                                                              'cartesian_position')
        block_level_0_orientation = self._stage.get_object_state('tool_level_0_col_0_row_0',
                                                                 'orientation')

        block_level_1_position = self._stage.get_object_state('tool_level_1_col_0_row_0',
                                                              'cartesian_position')
        block_level_1_orientation = self._stage.get_object_state('tool_level_1_col_0_row_0',
                                                                 'orientation')
        rewards = list()

        # term 1
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     self.previous_block_level_0_position)
        previous_distance_from_block = np.linalg.norm(
            self.previous_end_effector_positions -
            self.previous_block_level_0_position)
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)

        # term 2
        current_horizontal_distance_from_blocks = np.linalg.norm(block_level_0_position[:2] -
                                                                 block_level_1_position[:2])

        previous_horizontal_distance_from_blocks = np.linalg.norm(self.previous_block_level_0_position[:2] -
                                                                  self.previous_block_level_1_position[:2])

        rewards.append(previous_horizontal_distance_from_blocks -
                       current_horizontal_distance_from_blocks)

        # term 3
        current_vertical_distance_from_blocks = np.linalg.norm(block_level_0_position[2] -
                                                               block_level_1_position[2] - 0.065)

        previous_vertical_distance_from_blocks = np.linalg.norm(self.previous_block_level_0_position[2] -
                                                                self.previous_block_level_1_position[2] - 0.065)

        rewards.append(previous_vertical_distance_from_blocks -
                       current_vertical_distance_from_blocks)

        # term 4
        rewards.append(- np.linalg.norm(
            joint_velocities - self.previous_joint_velocities))

        update_task_info = {'current_end_effector_positions': end_effector_positions,
                            'current_velocity': joint_velocities,
                            'current_tool_level_0_col_0_row_0_position': block_level_0_position,
                            'current_tool_level_0_col_0_row_0_orientation': block_level_0_orientation,
                            'current_tool_level_1_col_0_row_0_position': block_level_1_position,
                            'current_tool_level_1_col_0_row_0_orientation': block_level_1_orientation
                            }
        return rewards, update_task_info

    def _set_task_state(self):
        """

        :return:
        """
        self.previous_end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_joint_velocities = \
            self._robot.get_latest_full_state()['velocities']
        self.previous_block_level_0_position = self._stage.get_object_state('tool_level_0_col_0_row_0',
                                                                            'cartesian_position')
        self.previous_block_level_0_orientation = self._stage.get_object_state('tool_level_0_col_0_row_0',
                                                                               'orientation')

        self.previous_block_level_1_position = self._stage.get_object_state('tool_level_1_col_0_row_0',
                                                                            'cartesian_position')
        self.previous_block_level_1_orientation = self._stage.get_object_state('tool_level_1_col_0_row_0',
                                                                               'orientation')

    def _update_task_state(self, update_task_info):
        """

        :param update_task_info:
        :return:
        """
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_tool_level_0_col_0_row_0_position = \
            update_task_info['current_tool_level_0_col_0_row_0_position']
        self.previous_tool_level_1_col_0_row_0_position = \
            update_task_info['current_tool_level_1_col_0_row_0_position']
        self.previous_tool_level_0_col_0_row_0_orientation = \
            update_task_info['current_tool_level_0_col_0_row_0_orientation']
        self.previous_tool_level_1_col_0_row_0_orientation = \
            update_task_info['current_tool_level_1_col_0_row_0_orientation']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return