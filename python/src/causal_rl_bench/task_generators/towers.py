from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
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
            kwargs.get("number_of_blocks_in_tower", np.array([2, 2, 2]))
        self.task_params["tower_dims"] = \
            kwargs.get("tower_dims", np.array([0.1, 0.1, 0.1]))
        self.current_tower_dims = self.task_params["tower_dims"]
        self.current_number_of_blocks_in_tower = \
            self.task_params["number_of_blocks_in_tower"]
        self.current_tool_block_mass = self.task_params["tool_block_mass"]

    def get_description(self):
        return "Task where the goal is to stack arbitrary number of towers side by side"

    #TODO: add obstacles interventions? up to a 10 obstacles?
    def _set_up_stage_arena(self):
        block_size = self.current_number_of_blocks_in_tower / self.task_params["tower_dims"]
        curr_height = 0.01 - block_size/2
        for level in range(self.task_params["number_of_blocks_in_tower"][-1]):
            curr_height += block_size
            curr_x = -self.task_params["tower_dims"][0]/2 - block_size
            curr_y = -self.task_params["tower_dims"][1]/ 2 - block_size
            for row in range(self.task_params["number_of_blocks_in_tower"][0]):
        default_start_position = -(number_of_blocks_per_level *
                                  self.task_params["blocks_min_size"])/2
        default_start_position += self.task_params["blocks_min_size"]/2
        curr_height = 0.01 - self.task_params["blocks_min_size"]/2
        change_per_level = 0.005
        rigid_block_side = 0.1
        for level in range(self.task_params["num_of_levels"]):
            change_per_level *= -1
            curr_height += self.task_params["blocks_min_size"]
            start_position = default_start_position + change_per_level
            rigid_block_side *= -1
            for i in range(number_of_blocks_per_level):
                self.stage.add_silhoutte_general_object(name="goal_"+"level_"+
                                                             str(level)+"_num_"+
                                                             str(i),
                                                        shape="cube",
                                                        position=[start_position,
                                                                  0, curr_height],
                                                        orientation=[0, 0, 0, 1],
                                                        size=np.repeat(self.task_params
                                                        ["blocks_min_size"], 3))
                self.task_stage_observation_keys.append("goal_"+"level_"+
                                                         str(level)+"_num_"+
                                                         str(i)+'_position')
                self.task_stage_observation_keys.append("goal_" + "level_" +
                                                        str(level) + "_num_" +
                                                        str(i) + '_orientation')
                self.stage.add_rigid_general_object(name="tool_" + "level_" +
                                                          str(level) + "_num_" +
                                                          str(i),
                                                    shape="cube",
                                                    position=[start_position,
                                                              rigid_block_side,
                                                              curr_height],
                                                    orientation=[0, 0, 0, 1],
                                                    size=np.repeat(self.task_params
                                                        ["blocks_min_size"], 3),
                                                    mass=self.task_params
                                                    ["tool_block_mass"])
                self.task_stage_observation_keys.append("tool_" + "level_" +
                                                        str(level) + "_num_" +
                                                        str(i) + '_position')
                self.task_stage_observation_keys.append("tool_" + "level_" +
                                                        str(level) + "_num_" +
                                                        str(i) + '_orientation')
                start_position += self.task_params["blocks_min_size"]
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return