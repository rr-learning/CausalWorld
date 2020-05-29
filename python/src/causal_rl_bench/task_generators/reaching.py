"""
causal_rl_bench.task_generators/reaching.py
===========================================
"""
from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np


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
        self.default_goal_60 = kwargs.get("default_goal_60",
                                          np.array([0, 0, 0.15]))
        self.default_goal_120 = kwargs.get("default_goal_120",
                                           np.array([0, 0, 0.2]))
        self.default_goal_300 = kwargs.get("default_goal_300",
                                           np.array([0, 0, 0.25]))
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        self.stage.add_silhoutte_general_object(name="goal_60",
                                                shape="sphere",
                                                color=np.array([1, 0, 0]),
                                                position=self.default_goal_60)
        self.stage.add_silhoutte_general_object(name="goal_120",
                                                shape="sphere",
                                                color=np.array([0, 1, 0]),
                                                position=self.default_goal_120)
        self.stage.add_silhoutte_general_object(name="goal_300",
                                                shape="sphere",
                                                color=np.array([0, 0, 1]),
                                                position=self.default_goal_300)
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a " \
            "point for each finger"

    def _reset_task(self):
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal, info):
        end_effector_positions_goal = desired_goal
        current_end_effector_positions = achieved_goal
        previous_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal -
            info['previous_end_effector_positions'])
        current_dist_to_goal = np.linalg.norm(end_effector_positions_goal
                                              - current_end_effector_positions)
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(-np.linalg.norm(info['current_torque']))
        rewards.append(-np.linalg.norm(np.abs(
            info['current_joint_velocities'] -
            info['previous_joint_velocities']), ord=2))
        return rewards

    def get_info(self):
        info = dict()
        info['current_end_effector_positions'] = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        info['previous_end_effector_positions'] = \
            self.previous_end_effector_positions
        info['current_torque'] = \
            self.robot.latest_full_state.torque
        info['current_joint_velocities'] = \
            self.robot.latest_full_state.velocity
        info['previous_joint_velocities'] = \
            self.previous_joint_velocities
        return info

    def _update_task_state(self, info):
        self.previous_end_effector_positions = \
            info['current_end_effector_positions']
        self.previous_joint_velocities = \
            info['current_joint_velocities']
        return





