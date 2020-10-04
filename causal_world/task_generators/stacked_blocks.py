from causal_world.task_generators.base_task import BaseTask
from causal_world.utils.rotation_utils import euler_to_quaternion
import numpy as np
import copy


class StackedBlocksGeneratorTask(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.08,
                 joint_positions=None,
                 blocks_min_size=0.035,
                 num_of_levels=5,
                 max_level_width=0.25):
        """
        This task generator will generate a task for stacking an arbitrary random
        configuration of blocks above each other.

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
        :param joint_positions: (nd.array) specifies the joints position to start
                                            the episode with. None if the default
                                            to be used.
        :param blocks_min_size: (float) specifies the blocks minimum size/
                                      side length for the goal shape generator.
        :param num_of_levels: (int) specifies the number of levels to be
                                    generated.
        :param max_level_width: (float) specifies the maximum width of the
                                        goal shape.
        """
        super().__init__(task_name="stacked_blocks",
                         variables_space=variables_space,
                         fractional_reward_weight=fractional_reward_weight,
                         dense_reward_weights=dense_reward_weights,
                         activate_sparse_reward=activate_sparse_reward)
        self._task_robot_observation_keys = ["time_left_for_task",
                                            "joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions"]
        self._task_params["tool_block_mass"] = tool_block_mass
        self._task_params["joint_positions"] = joint_positions
        self._task_params["blocks_min_size"] = blocks_min_size
        self._task_params["num_of_levels"] = num_of_levels
        self._task_params["max_level_width"] = max_level_width
        self.current_stack_levels = self._task_params["num_of_levels"]
        self.current_blocks_mass = self._task_params["tool_block_mass"]
        self.current_blocks_min_size = self._task_params["blocks_min_size"]
        self.current_max_level_width = self._task_params["max_level_width"]

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return "Task where the goal is to stack arbitrary shapes of cuboids"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        silhouettes_creation_dicts = []
        number_of_blocks_per_level = int(self._task_params["max_level_width"] \
                                         / self._task_params["blocks_min_size"])
        default_start_position = -(number_of_blocks_per_level *
                                   self._task_params["blocks_min_size"]) / 2
        default_start_position += self._task_params["blocks_min_size"] / 2
        curr_height = 0 \
                      - self._task_params["blocks_min_size"] / 2
        change_per_level = 0.005
        rigid_block_side = 0.1
        for level in range(self._task_params["num_of_levels"]):
            change_per_level *= -1
            curr_height += self._task_params["blocks_min_size"]
            start_position = default_start_position + change_per_level
            rigid_block_side *= -1
            for i in range(number_of_blocks_per_level):
                creation_dict = {
                    'name': "tool_" + "level_" + str(level) + "_num_" + str(i),
                    'shape': "cube",
                    'initial_position': [
                        start_position, rigid_block_side, curr_height
                    ],
                    'initial_orientation': [0, 0, 0, 1],
                    'size': np.repeat(self._task_params["blocks_min_size"], 3),
                    'mass': self._task_params["tool_block_mass"]
                }
                self._stage.add_rigid_general_object(**creation_dict)
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) + '_type')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) + '_size')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_cartesian_position')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_orientation')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_linear_velocity')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_angular_velocity')
                creation_dict = {
                    'name': "goal_" + "level_" + str(level) + "_num_" + str(i),
                    'shape': "cube",
                    'position': [start_position, 0, curr_height],
                    'orientation': [0, 0, 0, 1],
                    'size': np.repeat(self._task_params["blocks_min_size"], 3)
                }
                silhouettes_creation_dicts.append(copy.deepcopy(creation_dict))
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) + '_type')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) + '_size')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_cartesian_position')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level) + "_num_" +
                                                         str(i) +
                                                         '_orientation')
                start_position += self._task_params["blocks_min_size"]
        for i in range(len(silhouettes_creation_dicts)):
            self._stage.add_silhoutte_general_object(
                **silhouettes_creation_dicts[i])
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
        intervention_dict['stack_levels'] = np.\
            random.uniform(intervention_space['stack_levels'][0],
                           intervention_space['stack_levels'][1])
        intervention_dict['blocks_mass'] = np. \
            random.uniform(intervention_space['blocks_mass'][0],
                           intervention_space['blocks_mass'][1])
        intervention_dict['blocks_min_size'] = np. \
            random.uniform(intervention_space['blocks_min_size'][0],
                           intervention_space['blocks_min_size'][1])
        intervention_dict['max_level_width'] = np. \
            random.uniform(intervention_space['max_level_width'][0],
                           intervention_space['max_level_width'][1])
        return intervention_dict

    def get_task_generator_variables_values(self):
        """

        :return: (dict) specifying the variables belonging to the task itself.
        """
        return {
            'stack_levels': self.current_stack_levels,
            'blocks_mass': self.current_blocks_mass,
            'blocks_min_size': self.current_blocks_min_size,
            'max_level_width': self.current_max_level_width
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
        if "max_level_width" in interventions_dict:
            self.current_max_level_width = interventions_dict["max_level_width"]
        if "blocks_min_size" in interventions_dict:
            self.current_blocks_min_size = interventions_dict["blocks_min_size"]
        if "stack_levels" in interventions_dict:
            self.current_stack_levels = interventions_dict["stack_levels"]
        if "blocks_mass" in interventions_dict:
            self.current_blocks_mass = interventions_dict["blocks_mass"]
        if "max_level_width" in interventions_dict or "blocks_min_size" in \
                interventions_dict or \
                "stack_levels" in interventions_dict:
            self._create_new_challenge(
                num_of_levels=int(self.current_stack_levels),
                blocks_min_size=self.current_blocks_min_size,
                blocks_mass=self.current_blocks_mass,
                max_level_width=self.current_max_level_width)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self._stage._rigid_objects:
                if self._stage._rigid_objects[rigid_object].is_not_fixed():
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.current_blocks_mass
            self._stage.apply_interventions(new_interventions_dict)

        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_intervention_space_b()
        self._set_intervention_space_a()
        self._set_intervention_space_a_b()
        self._stage.finalize_stage()
        return True, reset_observation_space

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(StackedBlocksGeneratorTask, self).\
            _set_intervention_space_a()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_a[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_a[rigid_object]['size']
        self._intervention_space_a['stack_levels'] = \
            np.array([1, 5])
        self._intervention_space_a['blocks_mass'] = \
            np.array([0.02, 0.06])
        self._intervention_space_a['blocks_min_size'] = \
            np.array([0.035, 0.065])
        self._intervention_space_a['max_level_width'] = \
            np.array([0.035, 0.12])
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(StackedBlocksGeneratorTask, self).\
            _set_intervention_space_b()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_b[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_b[rigid_object]['size']
        self._intervention_space_b['stack_levels'] = \
            np.array([6, 8])
        self._intervention_space_b['blocks_mass'] = \
            np.array([0.06, 0.08])
        self._intervention_space_b['blocks_min_size'] = \
            np.array([0.065, 0.075])
        self._intervention_space_b['max_level_width'] = \
            np.array([0.12, 0.15])
        return

    def _create_new_challenge(self, num_of_levels, blocks_min_size, blocks_mass,
                              max_level_width):
        """

        :param num_of_levels:
        :param blocks_min_size:
        :param blocks_mass:
        :param max_level_width:

        :return:
        """
        silhouettes_creation_dicts = []
        self._stage.remove_everything()
        self._task_stage_observation_keys = []
        block_sizes, positions, chosen_y = self._generate_random_target(
            num_of_levels=num_of_levels,
            min_size=blocks_min_size,
            max_level_width=max_level_width)
        for level_num in range(len(block_sizes)):
            for i in range(len(block_sizes[level_num])):
                creation_dict = {
                    'name':
                        "tool_" + "level_" + str(level_num) + "_num_" + str(i),
                    'shape':
                        "cube",
                    'mass':
                        blocks_mass,
                    'color':
                        np.random.uniform(0, 1, size=[3]),
                    'size':
                        block_sizes[level_num][i]
                }
                self._stage.add_rigid_general_object(**creation_dict)
                block_position = self._stage.random_position(
                    height_limits=block_sizes[level_num][i][-1]/2.0)
                block_orientation = euler_to_quaternion(
                    [0, 0, np.random.uniform(-np.pi, np.pi)])
                self._stage.set_objects_pose(names=[
                    "tool_" + "level_" + str(level_num) + "_num_" + str(i)
                ],
                                             positions=[block_position],
                                             orientations=[block_orientation])
                trial_index = 0
                self._robot.step_simulation()
                while not self._stage.check_feasiblity_of_stage() and \
                        trial_index < 10:
                    block_position = self._stage.random_position(
                        height_limits=[block_sizes[level_num][i][-1]/2.0, 0.15])
                    block_orientation = euler_to_quaternion(
                        [0, 0, np.random.uniform(-np.pi, np.pi)])
                    self._stage.set_objects_pose(names=[
                        "tool_" + "level_" + str(level_num) + "_num_" + str(i)
                    ],
                                                 positions=[block_position],
                                                 orientations=[
                                                     block_orientation
                                                 ])
                    self._robot.step_simulation()
                    trial_index += 1
                silhouette_position = [
                    positions[level_num][i], chosen_y,
                    (level_num + 1) * blocks_min_size +
                    (-blocks_min_size / 2 + 0)
                ]
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_type')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_size')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_cartesian_position')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_orientation')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_linear_velocity')
                self._task_stage_observation_keys.append("tool_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_angular_velocity')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_type')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_size')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_cartesian_position')
                self._task_stage_observation_keys.append("goal_" + "level_" +
                                                         str(level_num) +
                                                         "_num_" + str(i) +
                                                         '_orientation')
                creation_dict = {
                    'name':
                        "goal_" + "level_" + str(level_num) + "_num_" + str(i),
                    'shape':
                        "cube",
                    'position':
                        np.array(silhouette_position),
                    'size':
                        np.array(block_sizes[level_num][i])
                }
                silhouettes_creation_dicts.append(copy.deepcopy(creation_dict))
        self.current_stack_levels = num_of_levels
        self.current_blocks_mass = blocks_mass
        self.current_blocks_min_size = blocks_min_size
        self.current_max_level_width = max_level_width
        for i in range(len(silhouettes_creation_dicts)):
            self._stage.add_silhoutte_general_object(
                **silhouettes_creation_dicts[i])
        return

    def _generate_random_block(self,
                               allowed_boundaries,
                               start_z,
                               min_size=0.035,
                               max_level_width=0.12):
        """
        This function will return a random position and size of a block
        while respecting the allowed boundaries passed

        :param allowed_boundaries:
        :param start_z:
        :param min_size:

        :return:
        """
        allowed_boundaries[0][0] = max(
            self._stage.get_arena_bb()[0][0] + min_size,
            allowed_boundaries[0][0])
        allowed_boundaries[1][0] = min(
            self._stage.get_arena_bb()[1][0] - min_size,
            allowed_boundaries[1][0])

        allowed_boundaries[0][1] = max(
            self._stage.get_arena_bb()[0][1] + min_size,
            allowed_boundaries[0][1])
        allowed_boundaries[1][1] = min(
            self._stage.get_arena_bb()[1][1] - min_size,
            allowed_boundaries[1][1])
        position_x_y = np.random.uniform(allowed_boundaries[0][:2],
                                         allowed_boundaries[1][:2])
        # choose size width, depth, height
        allowed_max_width = min(
            self._stage.get_arena_bb()[1][0] - position_x_y[0],
            position_x_y[0] - self._stage.get_arena_bb()[0][0]) * 2
        allowed_max_width = min(allowed_max_width, max_level_width)
        size = np.random.uniform(min_size,
                                 [allowed_max_width, min_size, min_size])
        position_z = start_z + size[-1] / 2
        position = np.array([position_x_y[0], position_x_y[1], position_z])
        return size, position

    def _generate_random_target(self,
                                num_of_levels=4,
                                min_size=0.035,
                                max_level_width=0.12):
        """
        This function generated a sampled target, should be modified to new
        sample goal

        :param levels_num:
        :param min_size:

        :return:
        """
        level_blocks = []
        current_boundaries = np.array([
            self._stage.get_arena_bb()[0][:2],
            self._stage.get_arena_bb()[1][:2]
        ])
        start_z = 0
        size, position = self._generate_random_block(
            allowed_boundaries=current_boundaries,
            start_z=start_z,
            min_size=min_size,
            max_level_width=max_level_width)
        level_blocks.append([[size[0], position[0]]])
        for level_index in range(1, num_of_levels):
            start_z = start_z + size[-1]
            new_allowed_boundaries = [
                position[:2] - size[:2] / 2, position[:2] + size[:2] / 2
            ]
            current_boundaries = [
                np.maximum(current_boundaries[0], new_allowed_boundaries[0]),
                np.minimum(current_boundaries[1], new_allowed_boundaries[1])
            ]
            size, position = self._generate_random_block(
                allowed_boundaries=current_boundaries,
                start_z=start_z,
                min_size=min_size,
                max_level_width=max_level_width)
            level_blocks.append([[size[0], position[0]]])
        chosen_y = position[1]
        new_level_blocks = \
            self._generate_blocks_to_use(level_blocks,
                                         min_size=min_size)
        new_level_blocks = \
            self._generate_blocks_to_use(new_level_blocks,
                                         min_size=min_size)
        new_level_blocks = \
            self._generate_blocks_to_use(new_level_blocks,
                                         min_size=min_size)
        #now return the actual block sizes that needs to be created
        block_sizes, positions = self._get_block_sizes(new_level_blocks,
                                                       min_size)

        return block_sizes, positions, chosen_y

    def _generate_blocks_to_use(self, level_blocks, min_size):
        """

        :param level_blocks:
        :param min_size:

        :return:
        """
        new_level_blocks = list(level_blocks)
        for i in range(len(level_blocks)):
            current_level_blocks = level_blocks[i]
            # try splitting randomly 4 times if the big block at
            # least has size of twice the min size
            for j in range(len(current_level_blocks)):
                if current_level_blocks[j][0] > min_size * 2:
                    # try splitting to half the size
                    # check for stability instead
                    block_1_center = current_level_blocks[j][1] + \
                                     current_level_blocks[j][0] / 4
                    block_1_size = current_level_blocks[j][0] / 2
                    block_2_size = current_level_blocks[j][0] - block_1_size
                    block_2_center = block_1_center - block_1_size / 2 - \
                                     block_2_size / 2
                    # now check for stability of
                    # structure with the previous levels
                    stability_levels_check = copy.deepcopy(new_level_blocks[:i +
                                                                            1])
                    stability_levels_check[i][j] = [
                        block_1_size, block_1_center
                    ]
                    stability_levels_check[i].append(
                        [block_2_size, block_2_center])
                    if self._is_stable_structure(stability_levels_check):
                        new_level_blocks[:i + 1] = stability_levels_check
        return new_level_blocks

    def _is_stable_structure(self, level_blocks):
        """

        :param level_blocks:

        :return:
        """
        # [[[width, center][width, center]]]
        current_min = -0.5
        current_max = 0.5
        for i in range(len(level_blocks)):
            current_level_blocks = level_blocks[i]
            new_min = 0.5
            new_max = -0.5
            # now check that for each block its
            # center lies within the current min and current max
            for block in current_level_blocks:
                new_min = min(new_min, block[1] - block[0] / 2)
                new_max = max(new_max, block[1] + block[0] / 2)
                if block[1] > current_max or block[1] < current_min:
                    return False
            current_min = new_min
            current_max = new_max
        return True

    def _get_block_sizes(self, level_blocks, min_size):
        """

        :param level_blocks:
        :param min_size:

        :return:
        """
        block_sizes = []
        positions = []
        for i in range(len(level_blocks)):
            block_sizes.append([])
            positions.append([])
            current_level_blocks = level_blocks[i]
            for block in current_level_blocks:
                block_sizes[-1].append([block[0], min_size, min_size])
                positions[-1].append(block[1])
        return block_sizes, positions
