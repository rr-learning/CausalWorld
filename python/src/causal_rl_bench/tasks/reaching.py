"""
causal_rl_bench/tasks/reaching.py
==================================
"""
from causal_rl_bench.tasks.base_task import BaseTask
import numpy as np
import math


class ReachingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="reaching")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions",
                                            "action_joint_positions",
                                            "end_effector_positions_goal"]
        self.task_params["sparse_reward_weight"] = \
            kwargs.get("sparse_reward_weight", 0)
        self.task_params["dense_reward_weights"] = \
            kwargs.get("dense_reward_weights", np.array([4, 4, 4, 4]))
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
        #TAKE care ur initial state needs to lie at the intersection of
        #both intervention sets
        self.initial_state['joint_positions'] = \
            self.robot.get_rest_pose()[0]
        self.initial_state['joint_velocities'] = \
            np.zeros([9,])
        # self.initial_state['goal_60'] = dict()
        # self.initial_state['goal_60']['position'] =\
        #     (self.stage.floor_inner_bounding_box[0] +
        #      self.stage.floor_inner_bounding_box[1]) / 2
        # self.initial_state['goal_120'] = dict()
        # self.initial_state['goal_120']['position'] =\
        #     (self.stage.floor_inner_bounding_box[0] +
        #      self.stage.floor_inner_bounding_box[1]) / 2
        # self.initial_state['goal_300'] = dict()
        # self.initial_state['goal_300']['position'] = \
        #     (self.stage.floor_inner_bounding_box[0] +
        #      self.stage.floor_inner_bounding_box[1]) / 2
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a point for each finger"

    def _set_up_non_default_observations(self):
        self._setup_non_default_robot_observation_key(
            observation_key="end_effector_positions_goal",
            observation_function=self._set_end_effector_positions_goal,
            lower_bound=[-0.5, -0.5, 0]*3,
            upper_bound=[0.5, 0.5, 0.5]*3)
        return

    def _set_end_effector_positions_goal(self):
        return list(self.stage.get_object_state('goal_60', 'position')) + \
               list(self.stage.get_object_state('goal_120', 'position')) + \
               list(self.stage.get_object_state('goal_300', 'position'))

    def _reset_task(self):
        if self.robot.latest_full_state is None:
            raise Exception("you are probably violating the intervention bounds"
                            "with you initial state")
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return

    def get_reward(self):
        end_effector_positions_goal = self._set_end_effector_positions_goal()
        current_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        previous_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal -
            self.previous_end_effector_positions)
        current_dist_to_goal = np.linalg.norm(end_effector_positions_goal
                                              - current_end_effector_positions)
        sparse_reward = self._compute_sparse_reward(
            achieved_goal=current_end_effector_positions,
            desired_goal=end_effector_positions_goal,
            info=self.get_info())
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(-np.linalg.norm(self.robot.latest_full_state.torque))
        rewards.append(-np.linalg.norm(np.abs(
            self.robot.latest_full_state.velocity -
            self.previous_joint_velocities), ord=2))
        reward = np.sum(np.array(rewards) * self.task_params["dense_reward_weights"]) \
                 + sparse_reward * self.task_params["sparse_reward_weight"]
        self.previous_end_effector_positions = current_end_effector_positions
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return reward

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




