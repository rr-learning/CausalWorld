from causal_world.task_generators.base_task import BaseTask
from causal_world.utils.rotation_utils import euler_to_quaternion
import numpy as np
import math


class GeneralGeneratorTask(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.08,
                 joint_positions=None,
                 tool_block_size=0.05,
                 nums_objects=5):
        """
        This task generator generates a general/ random configuration of the
        blocks by dropping random blocks from the air and waiting till it comes
        to a rest position and then this becomes the new shape/goal that the
        actor needs to achieve.

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
        :param tool_block_size: (float) specifies the blocks size.
        :param nums_objects: (int) specifies the number of objects to be dropped
                                   from the air.
        """
        super().__init__(task_name="general",
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
        self._task_params["nums_objects"] = nums_objects
        self._task_params["tool_block_size"] = tool_block_size
        self.default_drop_positions = [[0.1, 0.1, 0.2],
                                       [0, 0, 0.2],
                                       [0.05, 0.05, 0.3],
                                       [-0.05, -0.05, 0.1],
                                       [-0.12,  -0.12, 0.2],
                                       [-0.12, 0.12, 0.2],
                                       [0.12, -0.10, 0.3],
                                       [0.09, -0.08, 0.1]]
        self.tool_mass = self._task_params["tool_block_mass"]
        self.nums_objects = self._task_params["nums_objects"]
        self.tool_block_size = np.array(self._task_params["tool_block_size"])

    def get_description(self):
        """

       :return: (str) returns the description of the task itself.
        """
        return "Task where the goal is to rearrange " \
               "available objects into a target configuration"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        self._generate_goal_configuration_with_objects(default_bool=True)
        return

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(GeneralGeneratorTask, self)._set_intervention_space_a()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_a[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_a[rigid_object]['size']
        self._intervention_space_a['nums_objects'] = \
            np.array([1, 5])
        self._intervention_space_a['blocks_mass'] = \
            np.array([0.02, 0.06])
        self._intervention_space_a['tool_block_size'] = \
            np.array([0.05, 0.07])
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(GeneralGeneratorTask, self)._set_intervention_space_b()
        for visual_object in self._stage.get_visual_objects():
            del self._intervention_space_b[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._intervention_space_b[rigid_object]['size']
        self._intervention_space_b['nums_objects'] = \
            np.array([6, 9])
        self._intervention_space_b['blocks_mass'] = \
            np.array([0.06, 0.08])
        self._intervention_space_b['tool_block_size'] = \
            np.array([0.04, 0.05])
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
        intervention_dict['nums_objects'] = np. \
            random.randint(intervention_space['nums_objects'][0],
                           intervention_space['nums_objects'][1])
        intervention_dict['blocks_mass'] = np. \
            random.uniform(intervention_space['blocks_mass'][0],
                           intervention_space['blocks_mass'][1])
        intervention_dict['tool_block_size'] = np. \
            random.uniform(intervention_space['tool_block_size'][0],
                           intervention_space['tool_block_size'][1])
        return intervention_dict

    def get_task_generator_variables_values(self):
        """

        :return: (dict) specifying the variables belonging to the task itself.
        """
        return {
            'nums_objects': self.nums_objects,
            'blocks_mass': self.tool_mass,
            'tool_block_size': self.tool_block_size
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
        if "nums_objects" in interventions_dict:
            self.nums_objects = interventions_dict["nums_objects"]
        if "tool_block_size" in interventions_dict:
            self.tool_block_size = interventions_dict["tool_block_size"]
        if "blocks_mass" in interventions_dict:
            self.tool_mass = interventions_dict["blocks_mass"]
        if "nums_objects" in interventions_dict or "tool_block_size" in \
                interventions_dict:
            self._generate_goal_configuration_with_objects(default_bool=False)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self._stage.get_rigid_objects():
                if self._stage.get_rigid_objects()[rigid_object].is_not_fixed():
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.tool_mass
            self._stage.apply_interventions(new_interventions_dict)
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_intervention_space_b()
        self._set_intervention_space_a()
        self._set_intervention_space_a_b()
        self._stage.finalize_stage()
        return True, reset_observation_space

    def _generate_goal_configuration_with_objects(self, default_bool):
        """

        :param default_bool:

        :return:
        """
        self._stage.remove_everything()
        stage_low_bound = np.array(self._stage.get_arena_bb()[0])
        stage_low_bound[:2] += 0.04
        stage_upper_bound = np.array(self._stage.get_arena_bb()[1])
        stage_upper_bound[:2] -= 0.04
        stage_upper_bound[2] -= 0.08
        self._task_stage_observation_keys = []
        joint_positions = self._robot.get_joint_positions_raised()
        self._robot.reset_state(joint_positions=joint_positions,
                                joint_velocities=np.zeros(9))
        for object_num in range(self.nums_objects):
            if default_bool:
                dropping_position = self.default_drop_positions[
                    object_num % len(self.default_drop_positions)]
                dropping_orientation = [0, 0, 0, 1]
            else:
                dropping_position = np.random.uniform(
                    stage_low_bound,
                    stage_upper_bound)
                dropping_orientation = euler_to_quaternion(
                    np.random.uniform(low=0, high=2 * math.pi, size=3))
            creation_dict = {
                'name': "tool_" + str(object_num),
                'shape': "cube",
                'initial_position': dropping_position,
                'initial_orientation': dropping_orientation,
                'mass': self.tool_mass,
                'size': np.repeat(self.tool_block_size, 3)
            }
            self._stage.add_rigid_general_object(**creation_dict)
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_type')
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_size')
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_cartesian_position')
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_orientation')
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_linear_velocity')
            self._task_stage_observation_keys.append("tool_" + str(object_num) +
                                                     '_angular_velocity')
            self._robot.forward_simulation(time=0.8)
        for rigid_object in self._stage._rigid_objects:
            creation_dict = {
                'name':
                    rigid_object.replace('tool', 'goal'),
                'shape':
                    "cube",
                'position':
                    self._stage.get_object_state(rigid_object,
                                                 'cartesian_position'),
                'orientation':
                    self._stage.get_object_state(rigid_object, 'orientation'),
                'size':
                    np.repeat(self.tool_block_size, 3)
            }
            self._stage.add_silhoutte_general_object(**creation_dict)
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_type')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_size')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_cartesian_position')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_orientation')
            trial_index = 1
            block_position = self._stage.random_position(
                height_limits=[self.tool_block_size/2.0, 0.15])
            block_orientation = euler_to_quaternion(
                [0, 0, np.random.uniform(-np.pi, np.pi)])
            self._stage.set_objects_pose(names=[rigid_object],
                                         positions=[block_position],
                                         orientations=[block_orientation])
            self._robot.step_simulation()
            while not self._stage.check_feasiblity_of_stage() and \
                    trial_index < 10:
                block_position = self._stage.random_position(
                    height_limits=[self.tool_block_size/2.0, 0.15])
                block_orientation = euler_to_quaternion(
                    [0, 0, np.random.uniform(-np.pi, np.pi)])
                self._stage.set_objects_pose(names=[rigid_object],
                                             positions=[block_position],
                                             orientations=[block_orientation])
                self._robot.step_simulation()
                trial_index += 1
        self._robot.reset_state(joint_positions=joint_positions,
                                joint_velocities=np.zeros([
                                    9,
                                ]))
        self._robot.update_latest_full_state()
        return
