from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import quaternion_conjugate, \
    quaternion_mul
import numpy as np


class PushingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pushing",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 1),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([0, 0, 0])))
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
            kwargs.get("goal_block_position", np.array([0, 0., 0.0425]))
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
        return

    def _set_training_intervention_spaces(self):
        super(PushingTask, self)._set_training_intervention_spaces()
        for rigid_object in self.stage.rigid_objects:
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
        super(PushingTask, self)._set_testing_intervention_spaces()
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

    def _calculate_dense_rewards(self, desired_goal, achieved_goal, info):
        rewards = list()
        end_effector_positions = \
            info['current_end_effector_positions']

        # calculate first reward term
        current_distance_from_block = \
            np.linalg.norm(end_effector_positions -
                           info['current_tool_block_position'])
        previous_distance_from_block = np.linalg.norm(
            info['previous_end_effector_positions'] -
            info['previous_tool_block_position'])
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)

        # calculate second reward term
        previous_dist_to_goal = np.linalg.norm(info['goal_block_position'] -
                                               info['previous_tool_block_position'])
        current_dist_to_goal = np.linalg.norm(info['goal_block_position'] -
                                              info['current_tool_block_position'])
        rewards.append(previous_dist_to_goal - current_dist_to_goal)

        # calculate third reward term
        quat_diff_old = quaternion_mul(
            np.expand_dims(info['goal_block_orientation'], 0),
            quaternion_conjugate(np.expand_dims(
                info['previous_tool_block_orientation'], 0)))
        angle_diff_old = 2 * np.arccos(np.clip(quat_diff_old[:, 3], -1., 1.))

        quat_diff = quaternion_mul(np.expand_dims(info['goal_block_orientation'], 0),
                                   quaternion_conjugate(np.expand_dims(
                                       info['current_tool_block_orientation'],
                                       0)))
        current_angle_diff = 2 * np.arccos(np.clip(quat_diff[:, 3], -1., 1.))

        rewards.append(angle_diff_old[0] -
                       current_angle_diff[0])
        return rewards

    def _update_task_state(self, info):
        self.previous_end_effector_positions = \
            info['current_end_effector_positions']
        self.previous_object_position = \
            info['current_tool_block_position']
        self.previous_object_orientation = \
            info['current_tool_block_orientation']
        return

    def get_info(self):
        info = dict()
        info['current_end_effector_positions'] = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position).reshape(-1, 3)
        info['previous_end_effector_positions'] = \
            self.previous_end_effector_positions
        info['current_tool_block_position'] = \
            self.stage.get_object_state('tool_block',
                                        'position')
        info['previous_tool_block_position'] = \
            self.previous_object_position
        info['current_tool_block_orientation'] = \
            self.stage.get_object_state('tool_block',
                                        'orientation')
        info['previous_tool_block_orientation'] = \
            self.previous_object_orientation
        info['goal_block_position'] = \
            self.stage.get_object_state('goal_block',
                                        'position')
        info['goal_block_orientation'] = \
            self.stage.get_object_state('goal_block',
                                        'orientation')
        return info

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
