from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np
import math


class GeneralGeneratorTask(BaseTask):
    def __init__(self, **kwargs):
        """
        This task generator
        will most probably deal with camera data if u want to use the
        sample goal function
        :param kwargs:
        """
        super().__init__(task_name="general",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 1),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([])))
        self._task_robot_observation_keys = ["time_left_for_task",
                                            "joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]

        #for this task the stage observation keys will be set with the
        #goal/structure building
        self._task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.08)
        self._task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self._task_params["nums_objects"] = kwargs.get("nums_objects", 5)
        self._task_params["tool_block_size"] = \
            kwargs.get("tool_block_size", 0.05)
        self.default_drop_positions = [[0.1, 0.1, 0.2],
                                       [0, 0, 0.2],
                                       [0.05, 0.05, 0.3],
                                       [-0.05, -0.05, 0.1],
                                       [-0.12,  -0.12, 0.2],
                                       [-0.12, 0.12, 0.2],
                                       [0.12, -0.10, 0.3],
                                       [0.9, -0.8, 0.1]]
        self.tool_mass = self._task_params["tool_block_mass"]
        self.nums_objects = self._task_params["nums_objects"]
        self.tool_block_size = np.array(self._task_params["tool_block_size"])

    def get_description(self):
        """

        :return:
        """
        return "Task where the goal is to rearrange available objects into a target configuration"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        self.generate_goal_configuration_with_objects(default_bool=True)
        return

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
        # for now remove all possible interventions on the goal in general
        # intevrntions on size of objects might become tricky to handle
        # contradicting interventions here?
        super(GeneralGeneratorTask, self)._set_training_intervention_spaces()
        for visual_object in self._stage.get_visual_objects():
            del self._training_intervention_spaces[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._training_intervention_spaces[rigid_object]['size']
        self._training_intervention_spaces['nums_objects'] = \
            np.array([1, 15])
        self._training_intervention_spaces['blocks_mass'] = \
            np.array([0.02, 0.06])
        self._training_intervention_spaces['tool_block_size'] = \
            np.array([0.035, 0.08])
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        super(GeneralGeneratorTask, self)._set_testing_intervention_spaces()
        for visual_object in self._stage.get_visual_objects():
            del self._testing_intervention_spaces[visual_object]
        for rigid_object in self._stage.get_rigid_objects():
            del self._testing_intervention_spaces[rigid_object]['size']
        self._training_intervention_spaces['nums_objects'] = \
            np.array([15, 20])
        self._training_intervention_spaces['blocks_mass'] = \
            np.array([0.06, 0.08])
        self._training_intervention_spaces['tool_block_size'] = \
            np.array([0.045, 0.06])
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

        :return:
        """
        return {'nums_objects': self.nums_objects,
                'blocks_mass': self.tool_mass,
                'tool_block_size': self.tool_block_size}

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        # TODO: support level removal intervention
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
            self.generate_goal_configuration_with_objects(default_bool=False)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self._stage.get_rigid_objects():
                if self._stage.get_rigid_objects()[rigid_object].is_not_fixed:
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.tool_mass
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_testing_intervention_spaces()
        self._set_training_intervention_spaces()
        self._stage.finalize_stage()
        return True, reset_observation_space

    def generate_goal_configuration_with_objects(self, default_bool):
        """

        :param default_bool:
        :return:
        """
        #raise the fingers
        self._stage.remove_everything()
        self._task_stage_observation_keys = []
        joint_positions = self._robot.get_upper_joint_positions()
        self._robot.reset_state(joint_positions=joint_positions,
                                joint_velocities=np.zeros(9))
        # self.task_params["object_configs_list"] = []
        # self.rigid_objects_names = []
        for object_num in range(self.nums_objects):
            if default_bool:
                dropping_position = self.default_drop_positions[
                    object_num % len(self.default_drop_positions)]
                dropping_orientation = [0, 0, 0, 1]
            else:
                dropping_position = np.random.uniform(self._stage.get_arena_bb()[0],
                                                      self._stage.get_arena_bb()[1])
                dropping_orientation = euler_to_quaternion(np.random.uniform(
                    low=0, high=2 * math.pi, size=3))
            creation_dict = {'name': "tool_"+str(object_num),
                             'shape': "cube",
                             'initial_position': dropping_position,
                             'initial_orientation': dropping_orientation,
                             'mass': self.tool_mass,
                             'size': np.repeat(self.tool_block_size, 3)}
            self._stage.add_rigid_general_object(**creation_dict)
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_type')
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_size')
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_cartesian_position')
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_orientation')
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_linear_velocity')
            self._task_stage_observation_keys.append("tool_" +
                                                    str(object_num)
                                                    + '_angular_velocity')
            # turn on simulation for 0.5 seconds
            self._robot.forward_simulation(time=0.2)
        for rigid_object in self._stage._rigid_objects:
            #search for the rigid object in the creation list
            creation_dict = {'name': rigid_object.replace('tool', 'goal'),
                             'shape': "cube",
                             'position':
                                 self._stage.get_object_state(
                                     rigid_object, 'position'),
                             'orientation':
                                 self._stage.get_object_state(
                                     rigid_object, 'orientation'),
                             'size': np.repeat(self.tool_block_size, 3)}
            self._stage.add_silhoutte_general_object(**creation_dict)
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_type')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_size')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_cartesian_position')
            self._task_stage_observation_keys.append(
                rigid_object.replace('tool', 'goal') + '_orientation')
            #choose a random position for the rigid object now
            trial_index = 1
            block_position = self._stage.random_position(
                height_limits=[0.0425, 0.15])
            block_orientation = euler_to_quaternion(
                [0, 0, np.random.uniform(-np.pi, np.pi)])
            self._stage.set_objects_pose(names=[rigid_object],
                                         positions=[block_position],
                                         orientations=[block_orientation])
            self._robot.step_simulation()
            while not self._stage.check_feasiblity_of_stage() and \
                    trial_index < 10:
                block_position = self._stage.random_position(
                    height_limits=[0.0425, 0.15])
                block_orientation = euler_to_quaternion(
                    [0, 0, np.random.uniform(-np.pi, np.pi)])
                self._stage.set_objects_pose(names=[rigid_object],
                                             positions=[block_position],
                                             orientations=[block_orientation])
                self._robot.step_simulation()
                trial_index += 1
        self._robot.reset_state(
            joint_positions=self._robot.get_rest_pose()[0],
            joint_velocities=np.zeros([9, ]))
        self._robot.update_latest_full_state()
        return
