from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import quaternion_conjugate, \
    quaternion_mul, euler_to_quaternion
import numpy as np


class PushingTaskGenerator(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pushing",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 1),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([0, 10, 0])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        self.task_stage_observation_keys = ["tool_block_position",
                                            "tool_block_orientation",
                                            "goal_block_position",
                                            "goal_block_orientation"]

        self.task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.08)
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.initial_state["tool_block"] = dict()
        self.initial_state["tool_block"]["position"] = \
            kwargs.get("tool_block_position", np.array([0, 0, 0.0425]))
        self.initial_state["tool_block"]["orientation"] = \
            kwargs.get("tool_block_orientation", np.array([0, 0, 0, 1]))
        self.task_params["goal_block_position"] = \
            kwargs.get("goal_block_position", np.array([0, 0.10, 0.0425]))
        self.task_params["goal_block_orientation"] = \
            kwargs.get("goal_block_orientation", np.array([0, 0, 0, 1]))
        self.previous_end_effector_positions = None
        self.previous_object_position = None
        self.previous_object_orientation = None

    def get_description(self):
        return \
            "Task where the goal is to push " \
            "an object towards a goal position"

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="tool_block",
                                            shape="cube",
                                            mass=self.task_params[
                                                "tool_block_mass"],
                                            position=self.initial_state
                                            ["tool_block"]["position"],
                                            orientation=self.initial_state
                                            ["tool_block"]["orientation"])
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube",
                                                position=self.task_params[
                                                    "goal_block_position"],
                                                orientation=self.task_params[
                                                    "goal_block_orientation"])
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    def _set_training_intervention_spaces(self):
        super(PushingTaskGenerator, self)._set_training_intervention_spaces()
        for rigid_object in self.stage.rigid_objects:
            #TODO: make it a function of size
            self.training_intervention_spaces[rigid_object]['position'][0][-1] \
                = 0.0425
            self.training_intervention_spaces[rigid_object]['position'][1][-1] \
                = 0.0425
        for visual_object in self.stage.visual_objects:
            self.training_intervention_spaces[visual_object]['position'][0][-1] \
                = 0.0425
            self.training_intervention_spaces[visual_object]['position'][1][-1] \
                = 0.0425
        return

    def _set_testing_intervention_spaces(self):
        super(PushingTaskGenerator, self)._set_testing_intervention_spaces()
        for rigid_object in self.stage.rigid_objects:
            self.testing_intervention_spaces[rigid_object]['position'][0][-1] \
                = 0.0425
            self.testing_intervention_spaces[rigid_object]['position'][1][-1] \
                = 0.0425
        for visual_object in self.stage.visual_objects:
            self.testing_intervention_spaces[visual_object]['position'][0][-1] \
                = 0.0425
            self.testing_intervention_spaces[visual_object]['position'][1][-1] \
                = 0.0425
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        rewards = list()
        block_position = self.stage.get_object_state('tool_block',
                                                     'position')
        block_orientation = self.stage.get_object_state('tool_block',
                                                        'orientation')
        goal_position = self.stage.get_object_state('goal_block',
                                                    'position')
        goal_orientation = self.stage.get_object_state('goal_block',
                                                       'orientation')
        end_effector_positions = self.robot.compute_end_effector_positions(
            self.robot.latest_full_state.position)
        end_effector_positions = end_effector_positions.reshape(-1, 3)

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     block_position)
        previous_distance_from_block = np.linalg.norm(
            self.previous_end_effector_positions -
            self.previous_object_position)
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)

        # calculate second reward term
        previous_dist_to_goal = np.linalg.norm(goal_position -
                                               self.previous_object_position)
        current_dist_to_goal = np.linalg.norm(goal_position - block_position)
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        quat_diff_old = quaternion_mul(np.expand_dims(goal_orientation, 0),
                                       quaternion_conjugate(np.expand_dims(
                                           self.previous_object_orientation,
                                           0)))
        angle_diff_old = 2 * np.arccos(np.clip(quat_diff_old[:, 3], -1., 1.))

        quat_diff = quaternion_mul(np.expand_dims(goal_orientation, 0),
                                   quaternion_conjugate(np.expand_dims(
                                       block_orientation,
                                       0)))
        current_angle_diff = 2 * np.arccos(np.clip(quat_diff[:, 3], -1., 1.))

        rewards.append(angle_diff_old[0] -
                       current_angle_diff[0])
        update_task_info = {'current_end_effector_positions': end_effector_positions,
                            'current_tool_block_position': block_position,
                            'current_tool_block_orientation': block_orientation}
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_object_position = \
            update_task_info['current_tool_block_position']
        self.previous_object_orientation = \
            update_task_info['current_tool_block_orientation']
        return

    def _set_task_state(self):
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = \
            self.stage.get_object_state('tool_block', 'position')
        self.previous_object_orientation = \
            self.stage.get_object_state('tool_block', 'orientation')
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        #for example size on goal_or tool should be propagated to the other
        #TODO:if a goal block intervention would lead to change of sides then
        #change the other side as well?
        if 'goal_block' in interventions_dict:
            if 'size' in interventions_dict['goal_block']:
                if 'tool_block' not in interventions_dict:
                    interventions_dict['tool_block'] = dict()
                interventions_dict['tool_block']['size'] = \
                    interventions_dict['goal_block']['size']
        elif 'tool_block' in interventions_dict:
            if 'size' in interventions_dict['tool_block']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block']['size']
        return interventions_dict
