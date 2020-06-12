
from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np


class ReachingTaskGenerator(BaseTask):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
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
        self.current_number_of_obstacles = 0

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {'name': "goal_60",
                         'shape': "sphere",
                         'color': np.array([1, 0, 0]),
                         'position': self.task_params['default_goal_60']}
        self.stage.add_silhoutte_general_object(**creation_dict)
        self._creation_list.append([self.stage.add_silhoutte_general_object, creation_dict])
        creation_dict = {'name': "goal_120",
                         'shape': "sphere",
                         'color': np.array([0, 1, 0]),
                         'position': self.task_params['default_goal_120']}
        self.stage.add_silhoutte_general_object(**creation_dict)
        self._creation_list.append([self.stage.add_silhoutte_general_object, creation_dict])
        creation_dict = {'name': "goal_300",
                         'shape': "sphere",
                         'color': np.array([0, 0, 1]),
                         'position': self.task_params['default_goal_300']}
        self.stage.add_silhoutte_general_object(**creation_dict)
        self._creation_list.append([self.stage.add_silhoutte_general_object, creation_dict])
        self.task_stage_observation_keys = ["goal_60_position",
                                            "goal_120_position",
                                            "goal_300_position"]
        self.current_number_of_obstacles = 0
        self.initial_state['goal_60'] = dict()
        self.initial_state['goal_60']['position'] = self.task_params['default_goal_60']
        self.initial_state['goal_120'] = dict()
        self.initial_state['goal_120']['position'] = self.task_params['default_goal_120']
        self.initial_state['goal_300'] = dict()
        self.initial_state['goal_300']['position'] = self.task_params['default_goal_300']
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    def get_description(self):
        """

        :return:
        """
        return \
            "Task where the goal is to reach a " \
            "point for each finger"

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:
        :return:
        """
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
        """

        :param update_task_info:
        :return:
        """
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return

    def _set_task_state(self):
        """

        :return:
        """
        self.current_number_of_obstacles = 0
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return

    def get_desired_goal(self):
        """

        :return:
        """
        desired_goal = np.array([])
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_60', 'position'))
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_120', 'position'))
        desired_goal = np.append(desired_goal,
                                 self.stage.get_object_state('goal_300', 'position'))
        return desired_goal

    def get_achieved_goal(self):
        """

        :return:
        """
        achieved_goal = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        return np.array(achieved_goal)

    def _goal_distance(self, achieved_goal, desired_goal):
        """

        :param achieved_goal:
        :param desired_goal:
        :return:
        """
        current_end_effector_positions = achieved_goal
        current_dist_to_goal = np.abs(desired_goal -
                                      current_end_effector_positions)
        current_dist_to_goal_mean = np.mean(current_dist_to_goal)
        return np.array(current_dist_to_goal_mean)

    def _check_preliminary_success(self, goal_distance):
        """

        :param goal_distance:
        :return:
        """
        if goal_distance < 0.01:
            return True
        else:
            return False

    def get_info(self):
        """

        :return:
        """
        info = dict()
        info['possible_solution_intervention'] = dict()
        desired_goal = self.get_desired_goal()
        info['possible_solution_intervention']['joint_positions'] = \
            self.robot.get_joint_positions_from_tip_positions(desired_goal,
                                                              list(
                                                                  self.robot.latest_full_state.position))
        return info

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
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
        self.training_intervention_spaces['number_of_obstacles'] = \
            np.array([1, 5])

        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
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
        #TODO:dicuss this!
        self.testing_intervention_spaces['number_of_obstacles'] = \
            np.array([1, 5])
        return

    def get_task_generator_variables_values(self):
        """

        :return:
        """
        task_generator_variables = dict()
        task_generator_variables['number_of_obstacles'] = \
            self.current_number_of_obstacles
        return task_generator_variables

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        # TODO: support level removal intervention
        if len(interventions_dict) == 0:
            return True, False
        reset_observation_space = False
        if "number_of_obstacles" in interventions_dict:
            #if its more than what I have
            #TODO: maybe check feasibility of stage?
            if int(interventions_dict["number_of_obstacles"]) > self.current_number_of_obstacles:
                for i in range(self.current_number_of_obstacles, int(interventions_dict["number_of_obstacles"])):
                    self.stage.add_rigid_general_object(name="obstacle_"+str(i),
                                                        shape="static_cube",
                                                        size=
                                                        np.array([0.01, 0.01, 0.01]),
                                                        color=np.array([0, 0, 0]),
                                                        position=np.random.uniform(self.stage.floor_inner_bounding_box[0],
                                                                                   self.stage.floor_inner_bounding_box[1]))
                    self.current_number_of_obstacles += 1
            #TODO: if its less than what I have
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_testing_intervention_spaces()
        self._set_training_intervention_spaces()
        self.stage.finalize_stage()
        return True, reset_observation_space
