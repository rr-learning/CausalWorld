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
                                    np.array([100000,
                                              0, 0, 0])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions",
                                            "action_joint_positions"]
        self.task_stage_observation_keys = ["goal_60_position",
                                            "goal_120_position",
                                            "goal_300_position"]
        self.task_params['default_goal_60'] = kwargs.get("default_goal_60",
                                                         np.array([0, 0, 0.15]))
        self.task_params['default_goal_120'] = kwargs.get("default_goal_120",
                                                          np.array([0, 0, 0.2]))
        self.task_params['default_goal_300'] = kwargs.get("default_goal_300",
                                                          np.array([0, 0, 0.25]))
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        self.stage.add_silhoutte_general_object(name="goal_60",
                                                shape="sphere",
                                                color=np.array([1, 0, 0]),
                                                position=self.task_params['default_goal_60'])
        self.stage.add_silhoutte_general_object(name="goal_120",
                                                shape="sphere",
                                                color=np.array([0, 1, 0]),
                                                position=self.task_params['default_goal_120'])
        self.stage.add_silhoutte_general_object(name="goal_300",
                                                shape="sphere",
                                                color=np.array([0, 0, 1]),
                                                position=self.task_params['default_goal_300'])
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a " \
            "point for each finger"

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        end_effector_positions_goal = desired_goal
        current_end_effector_positions = achieved_goal
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

    def _set_task_state(self):
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return

    def get_desired_goal(self):
        desired_goal = np.array([])
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_60', 'position'))
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_120', 'position'))
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_300', 'position'))
        return desired_goal

    def get_achieved_goal(self):
        achieved_goal = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        return np.array(achieved_goal)

    def _goal_distance(self, achieved_goal, desired_goal):
        current_end_effector_positions = achieved_goal
        current_dist_to_goal = np.abs(desired_goal -
                                      current_end_effector_positions)
        current_dist_to_goal_mean = np.mean(current_dist_to_goal)
        return np.array(current_dist_to_goal_mean)

    def _check_preliminary_success(self, goal_distance):
        if goal_distance < 0.01:
            return True
        else:
            return False

    def get_info(self):
        info = dict()
        info['possible_solution_intervention'] = dict()
        desired_goal = self.get_desired_goal()
        info['possible_solution_intervention']['joint_positions'] = \
            self.robot.get_joint_positions_from_tip_positions(desired_goal,
                                                              list(
                                                                  self.robot.latest_full_state.position))
        return info

    def _set_training_intervention_spaces(self):
        # you can override these easily
        super(ReachingTaskGenerator,
              self)._set_training_intervention_spaces()
        lower_bound = np.array(self.stage.floor_inner_bounding_box[0])
        upper_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1/2 + \
                       self.stage.floor_inner_bounding_box[0]
        lower_bound[1] = float(upper_bound[1])
        upper_bound[1] = ((self.stage.floor_inner_bounding_box[1] -
                          self.stage.floor_inner_bounding_box[0]) * 3/4 + \
                          self.stage.floor_inner_bounding_box[0])[1]
        self.training_intervention_spaces['goal_60']['position'] = \
            np.array([lower_bound,
                      upper_bound]) #blue is finger 0, green 240
        lower_bound = np.array(self.stage.floor_inner_bounding_box[0])
        upper_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1 / 2 + \
                       self.stage.floor_inner_bounding_box[0]
        upper_bound[0] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 4 + \
                          self.stage.floor_inner_bounding_box[0])[1]
        self.training_intervention_spaces['goal_120']['position'] = \
            np.array([lower_bound,
                      upper_bound])  # blue is finger 0, green 240
        lower_bound = np.array(self.stage.floor_inner_bounding_box[0])
        upper_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1 / 2 + \
                       self.stage.floor_inner_bounding_box[0]
        upper_bound[1] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 4 +
                          self.stage.floor_inner_bounding_box[0])[1]
        self.training_intervention_spaces['goal_300']['position'] = \
            np.array([lower_bound,
                      upper_bound])

        return

    def _set_testing_intervention_spaces(self):
        super(ReachingTaskGenerator,
              self)._set_testing_intervention_spaces()
        lower_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1 / 2 + \
                       self.stage.floor_inner_bounding_box[0]
        lower_bound[0] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 3 / 4 +
                          self.stage.floor_inner_bounding_box[0])[1]
        upper_bound = np.array(self.stage.floor_inner_bounding_box[1])

        self.testing_intervention_spaces['goal_60']['position'] = \
            np.array([lower_bound,
                      upper_bound])
        lower_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1 / 2 + \
                       self.stage.floor_inner_bounding_box[0]
        lower_bound[0] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 4 +
                          self.stage.floor_inner_bounding_box[0])[1]
        upper_bound = np.array(self.stage.floor_inner_bounding_box[1])
        upper_bound[0] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 2 +
                          self.stage.floor_inner_bounding_box[0])[1]
        self.testing_intervention_spaces['goal_120']['position'] = \
            np.array([lower_bound,
                      upper_bound])
        lower_bound = (self.stage.floor_inner_bounding_box[1] -
                       self.stage.floor_inner_bounding_box[0]) * 1 / 2 + \
                      self.stage.floor_inner_bounding_box[0]
        lower_bound[1] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 4 +
                          self.stage.floor_inner_bounding_box[0])[1]
        upper_bound = np.array(self.stage.floor_inner_bounding_box[1])
        upper_bound[1] = ((self.stage.floor_inner_bounding_box[1] -
                           self.stage.floor_inner_bounding_box[0]) * 1 / 2 +
                          self.stage.floor_inner_bounding_box[0])[1]

        self.testing_intervention_spaces['goal_300']['position'] = \
            np.array([lower_bound,
                      upper_bound])
        return
