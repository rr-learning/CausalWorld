from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np


class TowersGeneratorTask(BaseTask):
    def __init__(self, **kwargs):
        """
        This task generator
        will most probably deal with camera data if u want to use the
        sample goal function
        :param kwargs:
        """
        super().__init__(task_name="towers",
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
        self.task_params["number_of_blocks_in_tower"] = \
            kwargs.get("number_of_blocks_in_tower", np.array([1, 1, 5]))
        self.task_params["tower_dims"] = \
            kwargs.get("tower_dims", np.array([0.035, 0.035, 0.175]))
        self.task_params["tower_center"] = \
            kwargs.get("tower_center", np.array([0, 0]))
        self.current_tower_dims = np.array(self.task_params["tower_dims"])
        self.current_number_of_blocks_in_tower = \
            np.array(self.task_params["number_of_blocks_in_tower"])
        self.current_tool_block_mass = int(self.task_params["tool_block_mass"])
        self.current_tower_center = np.array(self.task_params["tower_center"])

    def get_description(self):
        return "Task where the goal is to stack arbitrary number of towers side by side"

    #TODO: add obstacles interventions? up to a 10 obstacles?
    def _set_up_stage_arena(self):
        self._set_up_cuboid(self.current_tower_dims,
                            self.current_number_of_blocks_in_tower,
                            self.current_tower_center)
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    def _set_up_cuboid(self, tower_dims, number_of_blocks_in_tower, center_position):
        self.stage.remove_everything()
        self.task_stage_observation_keys = []
        block_size = tower_dims / number_of_blocks_in_tower
        curr_height = self.stage.floor_height - block_size[-1] / 2
        rigid_block_position = np.array([-0.12, -0.12, self.stage.floor_height + block_size[-1] / 2])
        for level in range(number_of_blocks_in_tower[-1]):
            curr_height += block_size[-1]
            curr_y = center_position[1] - tower_dims[1] / 2 - block_size[1] / 2
            for col in range(number_of_blocks_in_tower[1]):
                curr_y += block_size[1]
                curr_x = center_position[0] - tower_dims[0] / 2 - block_size[0] / 2
                for row in range(number_of_blocks_in_tower[0]):
                    curr_x += block_size[0]
                    self.stage.add_silhoutte_general_object(name="goal_" + "level_" +
                                                                 str(level) + "_col_" +
                                                                 str(col) + "_row_" + str(row),
                                                            shape="cube",
                                                            position=[curr_x, curr_y, curr_height],
                                                            orientation=[0, 0, 0, 1],
                                                            size=block_size)
                    self.stage.add_rigid_general_object(name="tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row),
                                                        shape="cube",
                                                        position=rigid_block_position,
                                                        orientation=[0, 0, 0, 1],
                                                        size=block_size,
                                                        mass=self.current_tool_block_mass)
                    self.task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_position')
                    self.task_stage_observation_keys.append("goal_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_orientation')
                    self.task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_position')
                    self.task_stage_observation_keys.append("tool_" + "level_" +
                                                             str(level) + "_col_" +
                                                             str(col) + "_row_" + str(row) + '_orientation')
                    rigid_block_position[:2] += block_size[:2]
                    rigid_block_position[:2] += 0.005
                    if np.any(rigid_block_position[:2] > np.array([0.12, 0.12])):
                        rigid_block_position[0] = -0.12
                        rigid_block_position[1] = -0.12
                        rigid_block_position[2] = rigid_block_position[2] + block_size[-1] / 2
        return

    def _set_training_intervention_spaces(self):
        #for now remove all possible interventions on the goal in general
        #intevrntions on size of objects might become tricky to handle
        #contradicting interventions here?
        super(TowersGeneratorTask, self)._set_training_intervention_spaces()
        for visual_object in self.stage.visual_objects:
            del self.training_intervention_spaces[visual_object]
        for rigid_object in self.stage.rigid_objects:
            del self.training_intervention_spaces[rigid_object]['size']
        self.training_intervention_spaces['number_of_blocks_in_tower'] = \
            np.array([[1, 1, 1], [4, 4,4]])
        self.training_intervention_spaces['blocks_mass'] = \
            np.array([0.02, 0.06])
        self.training_intervention_spaces['tower_dims'] = \
            np.array([[0.035, 0.035, 0.035], [0.10, 0.10, 0.10]])
        self.training_intervention_spaces['tower_center'] = \
            np.array([[-0.1, -0.1], [0.05, 0.05]])
        return

    def _set_testing_intervention_spaces(self):
        super(TowersGeneratorTask, self)._set_testing_intervention_spaces()
        for visual_object in self.stage.visual_objects:
            del self.testing_intervention_spaces[visual_object]
        for rigid_object in self.stage.rigid_objects:
            del self.testing_intervention_spaces[rigid_object]['size']
        self.testing_intervention_spaces['number_of_blocks_in_tower'] = \
            np.array([[4, 4, 4], [6, 6, ]])
        self.testing_intervention_spaces['blocks_mass'] = \
            np.array([0.06, 0.08])
        self.testing_intervention_spaces['tower_dims'] = \
            np.array([[0.10, 0.10, 0.10], [0.13, 0.13, 0.13]])
        self.testing_intervention_spaces['tower_center'] = \
            np.array([[0.05, 0.05], [0.1, 0.1]])
        return

    def sample_new_goal(self, training=True, level=None):
        intervention_dict = dict()
        if training:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        intervention_dict['number_of_blocks_in_tower'] = np. \
            random.randint(intervention_space['number_of_blocks_in_tower'][0],
                           intervention_space['number_of_blocks_in_tower'][1])
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
        return {'tower_dims': self.current_tower_dims,
                'blocks_mass': self.current_tool_block_mass,
                'number_of_blocks_in_tower': self.current_number_of_blocks_in_tower,
                'tower_center': self.current_tower_center}

    def apply_task_generator_interventions(self, interventions_dict):
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
        #TODO: tae care of center and orientation seperatly since we dont need to recreate everything,
        # just translate and rotate!!
        if "tower_dims" in interventions_dict or "number_of_blocks_in_tower" in interventions_dict or \
                "tower_center" in interventions_dict or "tower_orientation" in interventions_dict:
            self._set_up_cuboid(tower_dims=self.current_tower_dims,
                                number_of_blocks_in_tower=self.current_number_of_blocks_in_tower,
                                center_position=self.current_tower_center)
        elif "blocks_mass" in interventions_dict:
            new_interventions_dict = dict()
            for rigid_object in self.stage.rigid_objects:
                if self.stage.rigid_objects[rigid_object].is_not_fixed:
                    new_interventions_dict[rigid_object] = dict()
                    new_interventions_dict[rigid_object]['mass'] = \
                        self.current_tool_block_mass
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_testing_intervention_spaces()
        self._set_training_intervention_spaces()
        self.stage.finalize_stage()
        return True, reset_observation_space
