from causal_world.task_generators.base_task import BaseTask
import numpy as np
from causal_world.configs.world_constants import WorldConstants


class ReachingTaskGenerator(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([100000,0, 0, 0]),
                 default_goal_60=np.array([0, 0, 0.10]),
                 default_goal_120=np.array([0, 0, 0.13]),
                 default_goal_300=np.array([0, 0, 0.16]),
                 joint_positions=None,
                 activate_sparse_reward=False):
        """
        This task generator will generate a task for reaching.

         :param variables_space: (str) space to be used either 'space_a' or
                                      'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param default_goal_60: (nd.array) the position of the goal for first
                                           finger, x, y, z.
        :param default_goal_120: (nd.array) the position of the goal for second
                                           finger, x, y, z.
        :param default_goal_300: (nd.array) the position of the goal for third
                                           finger, x, y, z.
        :param joint_positions: (nd.array) specifies the joints position to start
                                            the episode with. None if the default
                                            to be used.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the mean distance
                                              from goal is < 0.01.
        """
        super().__init__(task_name="reaching",
                         variables_space=variables_space,
                         fractional_reward_weight=fractional_reward_weight,
                         dense_reward_weights=dense_reward_weights,
                         activate_sparse_reward=activate_sparse_reward)
        self._task_robot_observation_keys = ["time_left_for_task",
                                             "joint_positions",
                                             "joint_velocities",
                                             "end_effector_positions"]
        self._task_params['default_goal_60'] = default_goal_60
        self._task_params['default_goal_120'] = default_goal_120
        self._task_params['default_goal_300'] = default_goal_300
        self._task_params["joint_positions"] = joint_positions
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None
        self.current_number_of_obstacles = 0

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {
            'name': "goal_60",
            'shape': "sphere",
            'color': np.array([1, 0, 0]),
            'position': self._task_params['default_goal_60']
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        creation_dict = {
            'name': "goal_120",
            'shape': "sphere",
            'color': np.array([0, 1, 0]),
            'position': self._task_params['default_goal_120']
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        creation_dict = {
            'name': "goal_300",
            'shape': "sphere",
            'color': np.array([0, 0, 1]),
            'position': self._task_params['default_goal_300']
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        self._task_stage_observation_keys = [
            "goal_60_cartesian_position",
            "goal_120_cartesian_position",
            "goal_300_cartesian_position"
        ]
        return

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return \
            "Task where the goal is to reach a " \
            "goal point for each finger"

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        end_effector_positions_goal = desired_goal
        current_end_effector_positions = achieved_goal
        previous_dist_to_goal = np.linalg.norm(
            end_effector_positions_goal - self.previous_end_effector_positions)
        current_dist_to_goal = np.linalg.norm(end_effector_positions_goal -
                                              current_end_effector_positions)
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(
            -np.linalg.norm(self._robot.get_latest_full_state()['torques']))
        rewards.append(-np.linalg.norm(np.abs(self._robot.get_latest_full_state(
        )['velocities'] - self.previous_joint_velocities),
                                       ord=2))
        update_task_info = {
            'current_end_effector_positions':
                current_end_effector_positions,
            'current_velocity':
                self._robot.get_latest_full_state()['velocities']
        }
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
        self.previous_end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        self.previous_joint_velocities = \
            self._robot.get_latest_full_state()['velocities']
        return

    def get_desired_goal(self):
        """

        :return: (nd.array) specifies the desired goal as array of all three
                            positions of the finger goals.
        """
        desired_goal = np.array([])
        desired_goal = np.append(
            desired_goal,
            self._stage.get_object_state('goal_60', 'cartesian_position'))
        desired_goal = np.append(
            desired_goal,
            self._stage.get_object_state('goal_120', 'cartesian_position'))
        desired_goal = np.append(
            desired_goal,
            self._stage.get_object_state('goal_300', 'cartesian_position'))
        return desired_goal

    def get_achieved_goal(self):
        """

        :return: (nd.array) specifies the achieved goal as concatenated
                            end-effector positions.
        """
        achieved_goal = self._robot.get_latest_full_state(
        )['end_effector_positions']
        return np.array(achieved_goal)

    def _goal_reward(self, achieved_goal, desired_goal):
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

    def _check_preliminary_success(self, goal_reward):
        """

        :param goal_reward:

        :return:
        """
        if goal_reward < 0.01:
            return True
        else:
            return False

    def _calculate_fractional_success(self, goal_reward):
        """

        :param goal_reward:
        :return:
        """
        clipped_distance = np.clip(goal_reward, 0.01, 0.03)
        distance_from_success = clipped_distance - 0.01
        fractional_success = 1 - (distance_from_success / 0.02)
        return fractional_success

    def get_info(self):
        """

        :return: (dict) returns the info dictionary after every step of the
                        environment.
        """
        info = dict()
        info['desired_goal'] = self._current_desired_goal
        info['achieved_goal'] = self._current_achieved_goal
        info['success'] = self._task_solved
        if self._is_ground_truth_state_exposed:
            info['ground_truth_current_state_variables'] = \
                self.get_current_variable_values()
        if self._is_partial_solution_exposed:
            info['possible_solution_intervention'] = dict()
            info['possible_solution_intervention']['joint_positions'] = \
                self._robot.get_joint_positions_from_tip_positions(
                    self._current_desired_goal,
                    self._robot.get_latest_full_state()['positions'])
        info['fractional_success'] =\
            self._calculate_fractional_success(self._current_goal_reward)
        return info

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(ReachingTaskGenerator, self)._set_intervention_space_a()
        self._intervention_space_a['number_of_obstacles'] = \
            np.array([1, 5])

        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(ReachingTaskGenerator, self)._set_intervention_space_b()
        self._intervention_space_b['number_of_obstacles'] = \
            np.array([1, 5])
        return

    def get_task_generator_variables_values(self):
        """

        :return: (dict) specifying the variables belonging to the task itself.
        """
        task_generator_variables = dict()
        task_generator_variables['number_of_obstacles'] = \
            self.current_number_of_obstacles
        return task_generator_variables

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict: (dict) variables and their corresponding
                                   intervention value.

        :return: (tuple) first position if the intervention was successful or
                         not, and second position indicates if
                         observation_space needs to be reset.
        """
        if len(interventions_dict) == 0:
            return True, False
        reset_observation_space = False
        if "number_of_obstacles" in interventions_dict:
            if int(interventions_dict["number_of_obstacles"]
                  ) > self.current_number_of_obstacles:
                reset_observation_space = True
                for i in range(self.current_number_of_obstacles,
                               int(interventions_dict["number_of_obstacles"])):
                    self._stage.add_rigid_general_object(
                        name="obstacle_" + str(i),
                        shape="static_cube",
                        size=np.array([0.01, 0.01, 0.01]),
                        color=np.array([0, 0, 0]),
                        position=np.random.uniform(WorldConstants.ARENA_BB[0],
                                                   WorldConstants.ARENA_BB[1]))
                    self.current_number_of_obstacles += 1
                    self._task_stage_observation_keys.append("obstacle_" +
                                                             str(i) + "_type")
                    self._task_stage_observation_keys.append("obstacle_" +
                                                             str(i) + "_size")
                    self._task_stage_observation_keys.append(
                        "obstacle_" + str(i) + "_cartesian_position")
                    self._task_stage_observation_keys.append("obstacle_" +
                                                             str(i) +
                                                             "_orientation")
            else:
                return True, reset_observation_space
        else:
            raise Exception("this task generator variable "
                            "is not yet defined")
        self._set_intervention_space_b()
        self._set_intervention_space_a()
        self._set_intervention_space_a_b()
        self._stage.finalize_stage()
        return True, reset_observation_space
