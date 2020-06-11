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
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]

        #for this task the stage observation keys will be set with the
        #goal/structure building
        self.task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.08)
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.task_params["nums_objects"] = kwargs.get("nums_objects", 5)
        self.task_params["tool_block_size"] = \
            kwargs.get("tool_block_size", 0.05)
        self.default_drop_positions = [[0.1, 0.1, 0.2],
                                       [0, 0, 0.2],
                                       [0.05, 0.05, 0.3],
                                       [-0.05, -0.05, 0.1],
                                       [-0.12,  -0.12, 0.2],
                                       [-0.12, 0.12, 0.2],
                                       [0.12, -0.10, 0.3],
                                       [0.9, -0.8, 0.1]]
        self.tool_mass = self.task_params["tool_block_mass"]
        self.nums_objects = self.task_params["nums_objects"]
        self.tool_block_size = np.array(self.task_params["tool_block_size"])

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
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
        # for now remove all possible interventions on the goal in general
        # intevrntions on size of objects might become tricky to handle
        # contradicting interventions here?
        super(GeneralGeneratorTask, self)._set_training_intervention_spaces()
        for visual_object in self.stage.visual_objects:
            del self.training_intervention_spaces[visual_object]
        for rigid_object in self.stage.rigid_objects:
            del self.training_intervention_spaces[rigid_object]['size']
        self.training_intervention_spaces['nums_objects'] = \
            np.array([1, 15])
        self.training_intervention_spaces['blocks_mass'] = \
            np.array([0.02, 0.06])
        self.training_intervention_spaces['tool_block_size'] = \
            np.array([0.035, 0.08])
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        super(GeneralGeneratorTask, self)._set_testing_intervention_spaces()
        for visual_object in self.stage.visual_objects:
            del self.testing_intervention_spaces[visual_object]
        for rigid_object in self.stage.rigid_objects:
            del self.testing_intervention_spaces[rigid_object]['size']
        self.training_intervention_spaces['nums_objects'] = \
            np.array([15, 20])
        self.training_intervention_spaces['blocks_mass'] = \
            np.array([0.06, 0.08])
        self.training_intervention_spaces['tool_block_size'] = \
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
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
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
        if "nums_objects" in interventions_dict or "tool_block_size" in interventions_dict:
            self.generate_goal_configuration_with_objects(default_bool=False)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self.stage.rigid_objects:
                if self.stage.rigid_objects[rigid_object].is_not_fixed:
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.tool_mass
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_testing_intervention_spaces()
        self._set_training_intervention_spaces()
        self.stage.finalize_stage()
        return True, reset_observation_space

    def generate_goal_configuration_with_objects(self, default_bool):
        """

        :param default_bool:
        :return:
        """
        #raise the fingers
        self.stage.clear_memory()
        self.robot.tri_finger.reset_world()
        self.robot.clear()
        self.stage.clear()
        self.task_stage_observation_keys = []
        self._creation_list = []
        joint_positions = self.robot.robot_actions.joint_positions_upper_bounds
        self.robot.set_full_state(np.append(joint_positions,
                                            np.zeros(9)))
        # self.task_params["object_configs_list"] = []
        # self.rigid_objects_names = []
        for object_num in range(self.nums_objects):
            if default_bool:
                dropping_position = self.default_drop_positions[object_num % len(self.default_drop_positions)]
                dropping_orientation = [0, 0, 0, 1]
            else:
                dropping_position = np.random.uniform(self.stage.floor_inner_bounding_box[0],
                                                      self.stage.floor_inner_bounding_box[1])
                dropping_orientation = euler_to_quaternion(np.random.uniform(low=0, high=2 * math.pi, size=3))
            creation_dict = {'name': "tool_"+str(object_num),
                             'shape': "cube",
                             'position': dropping_position,
                             'orientation': dropping_orientation,
                             'mass': self.tool_mass,
                             'size': np.repeat(self.tool_block_size, 3)}
            self.stage.add_rigid_general_object(**creation_dict)
            self._creation_list.append([self.stage.add_rigid_general_object, creation_dict])
            self.task_stage_observation_keys.append("tool_" + str(object_num) + '_position')
            self.task_stage_observation_keys.append("tool_" + str(object_num) + '_orientation')
            # turn on simulation for 0.5 seconds
            self.robot.forward_simulation(time=0.2)
        for rigid_object in self.stage.rigid_objects:
            #search for the rigid object in the creation list
            rigid_object_creation_dict = None
            for created_object in self._creation_list:
                if created_object[1]['name'] == rigid_object:
                    rigid_object_creation_dict = created_object[1]
                    break
            creation_dict = {'name': rigid_object.replace('tool', 'goal'),
                             'shape': "cube",
                             'position': self.stage.get_object_state(rigid_object, 'position'),
                             'orientation': self.stage.get_object_state(rigid_object, 'orientation'),
                             'size': np.repeat(self.tool_block_size, 3)}
            self.stage.add_silhoutte_general_object(**creation_dict)
            self._creation_list.append([self.stage.add_silhoutte_general_object, creation_dict])
            self.task_stage_observation_keys.append(rigid_object.replace('tool', 'goal') + '_position')
            self.task_stage_observation_keys.append(rigid_object.replace('tool', 'goal') + '_orientation')
            #choose a random position for the rigid object now
            trial_index = 1
            block_position = self.stage.random_position(
                height_limits=[0.0425, 0.15])
            block_orientation = euler_to_quaternion(
                [0, 0, np.random.uniform(-np.pi, np.pi)])
            self.stage.set_objects_pose(names=[rigid_object],
                                        positions=[block_position],
                                        orientations=[block_orientation])
            rigid_object_creation_dict['position'] = block_position
            rigid_object_creation_dict['orientation'] = block_orientation
            self.stage.pybullet_client.stepSimulation()
            while not self.stage.check_feasiblity_of_stage() and \
                    trial_index < 10:
                block_position = self.stage.random_position(
                    height_limits=[0.0425, 0.15])
                block_orientation = euler_to_quaternion(
                    [0, 0, np.random.uniform(-np.pi, np.pi)])
                self.stage.set_objects_pose(names=[rigid_object],
                                            positions=[block_position],
                                            orientations=[block_orientation])
                rigid_object_creation_dict['position'] = block_position
                rigid_object_creation_dict['orientation'] = block_orientation
                self.stage.pybullet_client.stepSimulation()
                trial_index += 1
        return
