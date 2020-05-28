from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion, quaternion_conjugate, quaternion_mul
import numpy as np
import math


class PushingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pushing",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 0),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([1, 10, 1])))
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
            kwargs.get("goal_block_position", np.array([0.1, 0., 0.0425]))
        self.task_params["goal_block_orientation"] = \
            kwargs.get("goal_block_orientation", np.array([0, 0, 0, 1]))
        self.previous_end_effector_positions = None
        self.previous_object_position = None
        self.previous_object_orientation = None

    def get_description(self):
        return \
            "Task where the goal is to push " \
            "an object towards a goal position"

    def _set_training_intervention_spaces(self):
        self.training_intervention_spaces = dict()
        self.training_intervention_spaces['joint_positions'] = \
            np.array([[-math.radians(70), -math.radians(70),
                       -math.radians(160)] * 3,
                       [math.radians(40), -math.radians(20),
                        -math.radians(30)] * 3])
        self.training_intervention_spaces['tool_block'] = dict()
        self.training_intervention_spaces['goal_block'] = dict()
        self.training_intervention_spaces['tool_block']['size'] = \
            np.array([[0.0325, 0.0325, 0.0325], [0.065, 0.065, 0.065]])
        #TODO: we need to adapt this based on the size of the block chosen?
        #Don't know if we need to deal with this here?
        position_lower_bound = self.stage.floor_inner_bounding_box[0]
        position_upper_bound = (self.stage.floor_inner_bounding_box[0] +
                                self.stage.floor_inner_bounding_box[1]) / 2
        position_lower_bound[-1] = 0.0425
        position_upper_bound[-1] = 0.0425
        self.training_intervention_spaces['tool_block']['position'] = \
            np.array([position_lower_bound, position_upper_bound])
        self.training_intervention_spaces['goal_block']['size'] = \
            np.array([[0.0325, 0.0325, 0.0325], [0.065, 0.065, 0.065]])
        self.training_intervention_spaces['goal_block']['position'] = \
            np.array([position_lower_bound, position_upper_bound])
        return

    def _set_testing_intervention_spaces(self):
        self.testing_intervention_spaces = dict()
        self.testing_intervention_spaces['joint_positions'] = \
            np.array([[math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3,
                       [math.radians(70), 0,
                        math.radians(-2)] * 3])
        self.testing_intervention_spaces['tool_block'] = dict()
        self.testing_intervention_spaces['goal_block'] = dict()
        self.testing_intervention_spaces['tool_block']['size'] = \
            np.array([[0.065, 0.065, 0.065], [0.07, 0.07, 0.07]])
        position_lower_bound = (self.stage.floor_inner_bounding_box[0] +
                                self.stage.floor_inner_bounding_box[1]) / 2
        position_upper_bound = self.stage.floor_inner_bounding_box[1]
        position_lower_bound[-1] = 0.0425
        position_upper_bound[-1] = 0.0425
        self.testing_intervention_spaces['tool_block']['position'] = \
            np.array([position_lower_bound, position_upper_bound])
        self.testing_intervention_spaces['goal_block']['size'] = \
            np.array([[0.065, 0.065, 0.065], [0.07, 0.07, 0.07]])
        self.testing_intervention_spaces['goal_block']['position'] = \
            np.array([position_lower_bound, position_upper_bound])
        return

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
        return

    def _reset_task(self):
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

    def _calculate_dense_rewards(self):
        rewards = list()
        block_position = self.stage.get_object_state('tool_block',
                                                     'position')
        block_orientation = self.stage.get_object_state('tool_block',
                                                        'orientation')
        goal_position = self.stage.get_object_state('goal_block', 'position')
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

        # calculate third reward term
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
        update_task_info = {'end_effector_positions': end_effector_positions,
                            'object_position': block_position,
                            'object_orientation': block_orientation}
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        self.previous_end_effector_positions = \
            update_task_info['end_effector_positions']
        self.previous_object_position = \
            update_task_info['object_position']
        self.previous_object_orientation = \
            update_task_info['object_orientation']
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        #for example size on goal_or tool should be propagated to the other
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
