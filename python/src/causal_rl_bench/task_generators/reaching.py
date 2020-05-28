"""
causal_rl_bench.task_generators/reaching.py
==================================
"""
from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np
import math


class ReachingTaskGenerator(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="reaching",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 0),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([4, 4, 4, 4])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions",
                                            "action_joint_positions"]
        self.task_stage_observation_keys = ["goal_60_position",
                                            "goal_120_position",
                                            "goal_300_position"]
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        self.stage.add_silhoutte_general_object(name="goal_60",
                                                shape="sphere",
                                                color=np.array([1, 0, 0]),
                                                position=np.array([0, 0, 0.15]))
        self.stage.add_silhoutte_general_object(name="goal_120",
                                                shape="sphere",
                                                color=np.array([0, 1, 0]),
                                                position=np.array([0, 0, 0.2]))
        self.stage.add_silhoutte_general_object(name="goal_300",
                                                shape="sphere",
                                                color=np.array([0, 0, 1]),
                                                position=np.array([0, 0, 0.25]))
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a " \
            "point for each finger"

    def _set_training_intervention_spaces(self):
        self.training_intervention_spaces = dict()
        self.training_intervention_spaces['joint_positions'] = \
            np.array([[-math.radians(70), -math.radians(70),
                       -math.radians(160)] * 3,
                       [math.radians(40), -math.radians(20),
                        -math.radians(30)] * 3])
        self.training_intervention_spaces['goal_60'] = dict()
        self.training_intervention_spaces['goal_120'] = dict()
        self.training_intervention_spaces['goal_300'] = dict()
        self.training_intervention_spaces['goal_60']['position'] = \
            np.array([self.stage.floor_inner_bounding_box[0],
                      (self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2])
        self.training_intervention_spaces['goal_120']['position'] = \
            np.array([self.stage.floor_inner_bounding_box[0],
                      (self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2])
        self.training_intervention_spaces['goal_300']['position'] = \
            np.array([self.stage.floor_inner_bounding_box[0],
                      (self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2])
        return

    def _set_testing_intervention_spaces(self):
        self.testing_intervention_spaces = dict()
        self.testing_intervention_spaces['joint_positions'] = \
            np.array([[math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3,
                       [math.radians(70), 0,
                        math.radians(-2)] * 3])
        self.testing_intervention_spaces['goal_60'] = dict()
        self.testing_intervention_spaces['goal_120'] = dict()
        self.testing_intervention_spaces['goal_300'] = dict()
        self.testing_intervention_spaces['goal_60']['position'] = \
            np.array([(self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2,
                       self.stage.floor_inner_bounding_box[1]])
        self.testing_intervention_spaces['goal_120']['position'] = \
            np.array([(self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2,
                      self.stage.floor_inner_bounding_box[1]])
        self.testing_intervention_spaces['goal_300']['position'] = \
            np.array([(self.stage.floor_inner_bounding_box[0] +
                       self.stage.floor_inner_bounding_box[1]) / 2,
                      self.stage.floor_inner_bounding_box[1]])
        return

    def _reset_task(self):
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return

    def _calculate_dense_rewards(self):
        end_effector_positions_goal = self.get_desired_goal()
        current_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        previous_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal -
            self.previous_end_effector_positions)
        current_dist_to_goal = np.linalg.norm(end_effector_positions_goal
                                              - current_end_effector_positions)
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(-np.linalg.norm(self.robot.latest_full_state.torque))
        rewards.append(-np.linalg.norm(np.abs(
            self.robot.latest_full_state.velocity -
            self.previous_joint_velocities), ord=2))
        update_task_info = {'current_end_effector_positions':
                                current_end_effector_positions,
                            'current_velocity': np.copy(
                                self.robot.latest_full_state.velocity)}
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return





