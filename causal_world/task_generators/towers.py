from causal_world.task_generators.base_task import BaseTask
import numpy as np
import copy


class TowersGeneratorTask(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.08,
                 number_of_blocks_in_tower=np.array([1, 1, 5]),
                 tower_dims=np.array([0.035, 0.035, 0.175]),
                 tower_center=np.array([0, 0])):
        """
        This task generator will generate a task for stacking blocks into
        towers.
        :param variables_space: (str) space to be used either 'space_a' or
                                      'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the volumetric
                                              fraction overlap more than 90%.
        :param tool_block_mass: (float) specifies the blocks mass.
        :param number_of_blocks_in_tower: (nd.array) specifies the number of blocks
                                                     in the tower in each
                                                     direction x,y,z.
        :param tower_dims: (nd.array) (nd.array) specifies the dimension of
                                                 the tower in each
                                                 direction x,y,z.
        :param tower_center: (nd.array) specifies the cartesian position
                                               of the center of the tower,
                                               x, y, z.
        """
        super().__init__(task_name="towers",
                         variables_space=variables_space,
                         fractional_reward_weight=fractional_reward_weight,
                         dense_reward_weights=dense_reward_weights,
                         activate_sparse_reward=activate_sparse_reward)
        self._task_robot_observation_keys = ["time_left_for_task",
                                              "joint_positions",
                                              "joint_velocities",
                                              "end_effector_positions"]
        self._task_params["tower_dims"] = tower_dims
        self._task_params["tower_center"] = tower_center
        self._task_params["tool_block_mass"] = tool_block_mass
        self._task_params["number_of_blocks_in_tower"] = \
            number_of_blocks_in_tower
        self.current_tower_dims = np.array(self._task_params["tower_dims"])
        self.current_number_of_blocks_in_tower = \
            np.array(self._task_params["number_of_blocks_in_tower"])
        self.current_tool_block_mass = float(
            self._task_params["tool_block_mass"])
        self.current_tower_center = np.array(self._task_params["tower_center"])

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return "Task where the goal is to stack arbitrary number of " \
               "towers side by side"

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
        joint_positions = self._robot.get_joint_positions_raised()
        self._robot.set_full_state(np.append(joint_positions, np.zeros(9)))
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
                curr_x = center_position[
                    0] - tower_dims[0] / 2 - block_size[0] / 2
                for row in range(number_of_blocks_in_tower[0]):
                    curr_x += block_size[0]
                    creation_dict = {
                        'name':
                            "tool_" + "level_" + str(level) + "_col_" +
                            str(col) + "_row_" + str(row),
                        'shape':
                            "cube",
                        'initial_position':
                            np.copy(rigid_block_position),
                        'initial_orientation': [0, 0, 0, 1],
                        'mass':
                            self.current_tool_block_mass,
                        'size':
                            block_size
                    }
                    self._stage.add_rigid_general_object(**creation_dict)
                    creation_dict = {
                        'name':
                            "goal_" + "level_" + str(level) + "_col_" +
                            str(col) + "_row_" + str(row),
                        'shape':
                            "cube",
                        'position': [curr_x, curr_y, curr_height],
                        'orientation': [0, 0, 0, 1],
                        'size':
                            block_size
                    }
                    silhouettes_creation_dicts.append(
                        copy.deepcopy(creation_dict))
                    self._task_stage_observation_keys.append("tool_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) + '_type')
                    self._task_stage_observation_keys.append("tool_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) + '_size')
                    self._task_stage_observation_keys.append(
                        "tool_" + "level_" + str(level) + "_col_" + str(col) +
                        "_row_" + str(row) + '_cartesian_position')
                    self._task_stage_observation_keys.append("tool_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) +
                                                             '_orientation')
                    self._task_stage_observation_keys.append("tool_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) +
                                                             '_linear_velocity')
                    self._task_stage_observation_keys.append(
                        "tool_" + "level_" + str(level) + "_col_" + str(col) +
                        "_row_" + str(row) + '_angular_velocity')
                    self._task_stage_observation_keys.append("goal_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) + '_type')
                    self._task_stage_observation_keys.append("goal_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) + '_size')
                    self._task_stage_observation_keys.append(
                        "goal_" + "level_" + str(level) + "_col_" + str(col) +
                        "_row_" + str(row) + '_cartesian_position')
                    self._task_stage_observation_keys.append("goal_" +
                                                             "level_" +
                                                             str(level) +
                                                             "_col_" +
                                                             str(col) +
                                                             "_row_" +
                                                             str(row) +
                                                             '_orientation')
                    rigid_block_position[:2] += block_size[:2]
                    rigid_block_position[:2] += 0.005
                    if np.any(
                            rigid_block_position[:2] > np.array([0.12, 0.12])):
                        rigid_block_position[0] = -0.12
                        rigid_block_position[1] = -0.12
                        rigid_block_position[
                            2] = rigid_block_position[2] + block_size[-1] / 2

        for i in range(len(silhouettes_creation_dicts)):
            self._stage.add_silhoutte_general_object(
                **silhouettes_creation_dicts[i])
        return

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(TowersGeneratorTask, self)._set_intervention_space_a()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_a[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_a[rigid_object]['size']
        self._intervention_space_a['number_of_blocks_in_tower'] = \
            np.array([[1, 1, 1], [4, 4, 4]])
        self._intervention_space_a['blocks_mass'] = \
            np.array([0.02, 0.06])
        self._intervention_space_a['tower_dims'] = \
            np.array([[0.08, 0.08, 0.08], [0.12, 0.12, 0.12]])
        self._intervention_space_a['tower_center'] = \
            np.array([[-0.1, -0.1], [0.05, 0.05]])
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(TowersGeneratorTask, self)._set_intervention_space_b()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_b[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_b[rigid_object]['size']
        self._intervention_space_b['number_of_blocks_in_tower'] = \
            np.array([[4, 4, 4], [6, 6, 6]])
        self._intervention_space_b['blocks_mass'] = \
            np.array([0.06, 0.08])
        self._intervention_space_b['tower_dims'] = \
            np.array([[0.12, 0.12, 0.12], [0.20, 0.20, 0.20]])
        self._intervention_space_b['tower_center'] = \
            np.array([[0.05, 0.05], [0.1, 0.1]])
        return

    def sample_new_goal(self, level=None):
        """
        Used to sample new goal from the corresponding shape families.

        :param level: (int) specifying the level - not used for now.

        :return: (dict) the corresponding interventions dict that could then
                       be applied to get a new sampled goal.
        """
        intervention_dict = dict()
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_b
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
        intervention_dict['number_of_blocks_in_tower'] = [np. \
            random.randint(intervention_space['number_of_blocks_in_tower'][0][i],
                           intervention_space['number_of_blocks_in_tower'][1][i])
                                                          for i in range(3)]
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

        :return: (dict) specifying the variables belonging to the task itself.
        """
        return {
            'tower_dims': self.current_tower_dims,
            'blocks_mass': self.current_tool_block_mass,
            'number_of_blocks_in_tower': self.current_number_of_blocks_in_tower,
            'tower_center': self.current_tower_center
        }

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict: (dict) variables and their corresponding
                                   intervention value.

        :return: (tuple) first position if the intervention was successful or
                         not, and second position indicates if
                         observation_space needs to be reset.
        """
        if len(interventions_dict) == 0:
            return True, False
        reset_observation_space = True
        if "tower_dims" in interventions_dict:
            self.current_tower_dims = interventions_dict["tower_dims"]
        if "number_of_blocks_in_tower" in interventions_dict:
            self.current_number_of_blocks_in_tower = interventions_dict[
                "number_of_blocks_in_tower"]
        if "blocks_mass" in interventions_dict:
            self.current_tool_block_mass = interventions_dict["blocks_mass"]
        if "tower_center" in interventions_dict:
            self.current_tower_center = interventions_dict["tower_center"]
        if "tower_dims" in interventions_dict or "number_of_blocks_in_tower" in \
                interventions_dict or \
                "tower_center" in interventions_dict or "tower_orientation" in \
                interventions_dict:
            self._set_up_cuboid(tower_dims=self.current_tower_dims,
                                number_of_blocks_in_tower=self.
                                current_number_of_blocks_in_tower,
                                center_position=self.current_tower_center)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self._stage.get_rigid_objects():
                if self._stage.get_rigid_objects()[rigid_object].is_not_fixed():
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.current_tool_block_mass
            self._stage.apply_interventions(new_interventions_dict)
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_intervention_space_b()
        self._set_intervention_space_a()
        self._set_intervention_space_a_b()
        self._stage.finalize_stage()
        return True, reset_observation_space
