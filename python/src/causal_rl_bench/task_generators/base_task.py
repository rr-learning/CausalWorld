import numpy as np
import math
import copy
from causal_rl_bench.utils.state_utils import get_bounding_box_area
from causal_rl_bench.utils.state_utils import get_intersection
from causal_rl_bench.utils.rotation_utils import cart2cyl
import pybullet


class BaseTask(object):
    def __init__(self, task_name, intervention_split,
                 training, sparse_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 is_goal_distance_dense=True,
                 calculate_additional_dense_rewards=True):
        """

        :param task_name:
        :param intervention_split:
        :param training:
        :param sparse_reward_weight:
        :param dense_reward_weights:
        :param is_goal_distance_dense:
        :param calculate_additional_dense_rewards:
        """
        self.robot = None
        self.stage = None
        self.task_solved = False
        self.task_name = task_name
        self.task_robot_observation_keys = []
        self.task_stage_observation_keys = []
        # the helper keys are observations that are not included in the task observations but it will be needed in reward
        # calculation or new observations calculation
        self._robot_observation_helper_keys = []
        self._stage_observation_helper_keys = []
        self._non_default_robot_observation_funcs = dict()
        self._non_default_stage_observation_funcs = dict()
        self.current_full_observations_dict = dict()
        self.task_params = dict()
        self.task_params["task_name"] = self.task_name
        self.task_params["sparse_reward_weight"] = sparse_reward_weight
        self.task_params["dense_reward_weights"] = dense_reward_weights
        self.time_steps_elapsed_since_success = 0
        self.task_params['time_threshold_in_goal_state_secs'] = 0.1
        self.current_time_secs = 0
        self.training_intervention_spaces = dict()
        self.testing_intervention_spaces = dict()
        self.finished_episode = False
        self.task_params['intervention_split'] = intervention_split
        self.task_params['training'] = training
        self.task_params["joint_positions"] = None
        self.task_params['is_goal_distance_dense'] = is_goal_distance_dense
        self.task_params['calculate_additional_dense_rewards'] = \
            calculate_additional_dense_rewards
        self._current_starting_state = dict()
        self._default_starting_state = dict()
        self._empty_stage = None
        self._current_desired_goal = None
        self._current_achieved_goal = None
        self._current_goal_distance = None
        return

    def _save_pybullet_state(self):
        pybullet_state = dict()
        if self.stage._pybullet_client_full_id is not None:
                pybullet_state['full'] = pybullet.\
                    saveState(physicsClientId=self.stage._pybullet_client_full_id)
        if self.stage._pybullet_client_w_goal_id is not None:
                pybullet_state['w_goal'] = pybullet.\
                    saveState(physicsClientId=self.stage._pybullet_client_w_goal_id)
        if self.stage._pybullet_client_w_o_goal_id is not None:
            pybullet_state['w_o_goal'] = pybullet. \
                saveState(physicsClientId=self.stage._pybullet_client_w_o_goal_id)
        return pybullet_state

    def _restore_pybullet_state(self, pybullet_state):
        if self.stage._pybullet_client_full_id is not None:
                pybullet.\
                    restoreState(pybullet_state['full'],
                                 physicsClientId=self.stage._pybullet_client_full_id)
        if self.stage._pybullet_client_w_goal_id is not None:
                pybullet.\
                    restoreState(pybullet_state['w_goal'],
                                 physicsClientId=self.stage._pybullet_client_w_goal_id)
        if self.stage._pybullet_client_w_o_goal_id is not None:
            pybullet. \
                restoreState(pybullet_state['w_o_goal'],
                             physicsClientId=self.stage._pybullet_client_w_o_goal_id)
        return

    def _save_state(self):
        state = dict()
        state['stage_object_state'] = \
            self.stage.get_full_env_state()
        state['robot_object_state'] = \
            self.robot.get_full_env_state()
        state['task_observations'] = \
            copy.deepcopy(self.task_stage_observation_keys)
        return state

    def _restore_state(self, state_dict):
        #remove everything in the arena
        #old number of rigid objects and number of visual objects
        old_number_of_rigid_objects = len(self.stage.get_rigid_objects())
        old_number_of_visual_objects = len(self.stage.get_visual_objects())
        reset_observation_space = False
        self.stage.remove_everything()
        self._restore_pybullet_state(self._empty_stage)
        self.robot.set_full_env_state(state_dict
                                      ['robot_object_state'])
        self.stage.set_full_env_state(
            state_dict['stage_object_state'])
        new_number_of_rigid_objects = len(self.stage.get_rigid_objects())
        new_number_of_visual_objects = len(self.stage.get_visual_objects())
        if old_number_of_rigid_objects != new_number_of_rigid_objects:
            reset_observation_space = True
        if old_number_of_visual_objects != new_number_of_visual_objects:
            reset_observation_space = True
        self.task_stage_observation_keys = state_dict['task_observations']
        return reset_observation_space

    def is_in_training_mode(self):
        """

        :return:
        """
        if self.task_params['intervention_split'] and \
                self.task_params['training']:
            return True
        else:
            return False

    def set_super_sparse_reward(self):
        """

        :return:
        """
        self.task_params['is_goal_distance_dense'] = True

    def get_description(self):
        """

        :return:
        """
        return

    def _set_task_state(self):
        """

        :return:
        """
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        # handle the contradictory intervention
        # that changes each other (objects -> silhouettes)
        # and other way around sometimes
        return interventions_dict

    def _set_up_stage_arena(self):
        """

        :return:
        """
        return

    def _set_up_non_default_observations(self):
        """

        :return:
        """
        return

    def get_task_generator_variables_values(self):
        """

        :return:
        """
        return {}

    def apply_task_generator_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        return True, False

    def get_info(self):
        """

        :return:
        """
        info = dict()
        info['desired_goal'] = self._current_desired_goal
        info['achieved_goal'] = self._current_achieved_goal
        info['possible_solution_intervention'] = dict()
        for rigid_object in self.stage._rigid_objects:
            #check if there is an equivilant visual object corresponding
            possible_corresponding_goal = rigid_object.replace('tool', 'goal')
            if possible_corresponding_goal in self.stage._visual_objects:
                info['possible_solution_intervention'][rigid_object] = dict()
                info['possible_solution_intervention'][rigid_object]['position'] = \
                    self.stage.get_object_state(possible_corresponding_goal, 'position')
                info['possible_solution_intervention'][rigid_object]['orientation'] = \
                    self.stage.get_object_state(possible_corresponding_goal, 'orientation')
        info['ground_truth_scm_variables_values'] = self.get_current_scm_values()
        info['fractional_success'] = self._current_goal_distance
        return info

    def _update_task_state(self, update_task_state_dict):
        """

        :param update_task_state_dict:
        :return:
        """
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:
        :return:
        """
        return np.array([]), None

    def sample_new_goal(self, training=True, level=None):
        """

        :param training:
        :param level:
        :return:
        """
        #TODO: for now we just vary position as a new goal
        #Need to generalize this
        intervention_dict = dict()
        if training:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        for visual_object in self.stage._visual_objects:
            if visual_object in intervention_space and \
                    'position' in intervention_space[visual_object]:
                intervention_dict[visual_object] = dict()
                intervention_dict[visual_object]['position'] = \
                    self.stage.random_position(
                        height_limits=intervention_space[visual_object]['position'][:, 2],
                        radius_limits=intervention_space[visual_object]['position'][:, 0],
                        angle_limits=intervention_space[visual_object]['position'][:, 1])
        return intervention_dict

    # def reset_default_state(self):
    #     """
    #
    #     :return:
    #     """
    #     self.stage.remove_everything()
    #     self.task_stage_observation_keys = []
    #     self.robot.tri_finger.reset_world()
    #     self.task_stage_observation_keys = []
    #     self.initial_state = dict(self.default_state)
    #     self._set_up_stage_arena()
    #     self._set_testing_intervention_spaces()
    #     self._set_training_intervention_spaces()
    #     self.stage.finalize_stage()

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
        #you can override these easily
        self.training_intervention_spaces = dict()
        self.training_intervention_spaces['joint_positions'] = \
            np.array([[-math.radians(70), -math.radians(70),
                       -math.radians(160)] * 3,
                      [math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3])
        #any goal or object in arena put the position
        #and orientation modification
        for rigid_object in self.stage._rigid_objects:
            self.training_intervention_spaces[rigid_object] = dict()
            self.training_intervention_spaces[rigid_object]['position'] = \
                np.array([self.stage._floor_inner_bounding_box[0],
                          (self.stage._floor_inner_bounding_box[1] -
                           self.stage._floor_inner_bounding_box[0]) * 1 / 2 + \
                          self.stage._floor_inner_bounding_box[0]])
            self.training_intervention_spaces[rigid_object]['size'] = \
                np.array([[0.035, 0.035, 0.035], [0.065, 0.065, 0.065]])
# =======
#                 np.array([[0.0, - math.pi, self.stage.floor_height], [0.09, math.pi, 0.3]])
#             if self.stage.rigid_objects[rigid_object].__class__.__name__ == 'Cuboid':
#                 self.training_intervention_spaces[rigid_object]['size'] = \
#                     np.array([[0.035, 0.035, 0.035], [0.065, 0.065, 0.065]])
# >>>>>>> origin/master
            self.training_intervention_spaces[rigid_object]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
            self.training_intervention_spaces[rigid_object]['mass'] = \
                np.array([0.05, 0.1])
        for visual_object in self.stage._visual_objects:
            self.training_intervention_spaces[visual_object] = dict()
            self.training_intervention_spaces[visual_object]['position'] = \
                np.array([self.stage._floor_inner_bounding_box[0],
                          (self.stage._floor_inner_bounding_box[1] -
                           self.stage._floor_inner_bounding_box[0]) * 1 / 2 + \
                          self.stage._floor_inner_bounding_box[0]])
            self.training_intervention_spaces[visual_object]['size'] = \
                np.array([[0.035, 0.035, 0.035], [0.065, 0.065, 0.065]])
# =======
#                 np.array([[0.0, - math.pi, self.stage.floor_height], [0.09, math.pi, 0.15]])
#             if self.stage.visual_objects[visual_object].__class__.__name__ == 'SCuboid':
#                 self.training_intervention_spaces[visual_object]['size'] = \
#                     np.array([[0.035, 0.035, 0.035], [0.065, 0.065, 0.065]])
# >>>>>>> origin/master
            self.training_intervention_spaces[visual_object]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self.training_intervention_spaces['floor_color'] = \
            np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self.training_intervention_spaces['stage_color'] = \
            np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self.training_intervention_spaces['floor_friction'] = \
            np.array([0.3, 0.8])
        for link in self.robot.get_link_names():
            self.training_intervention_spaces[link] = dict()
            self.training_intervention_spaces[link]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            self.training_intervention_spaces[link]['mass'] = \
                np.array([0.2, 0.6])
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        # you can override these easily
        self.testing_intervention_spaces = dict()
        self.testing_intervention_spaces['joint_positions'] = \
            np.array([[math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3,
                      [math.radians(70), 0,
                       math.radians(-2)] * 3])
        # any goal or object in arena put the position
        # and orientation modification
        for rigid_object in self.stage._rigid_objects:
            self.testing_intervention_spaces[rigid_object] = dict()
            self.testing_intervention_spaces[rigid_object]['position'] = \
                np.array([(self.stage._floor_inner_bounding_box[1] -
                           self.stage._floor_inner_bounding_box[0]) * 1 / 2 + \
                          self.stage._floor_inner_bounding_box[0],
                          self.stage._floor_inner_bounding_box[1]])
            self.testing_intervention_spaces[rigid_object]['size'] = \
                np.array([[0.065, 0.065, 0.065], [0.075, 0.075, 0.075]])
# =======
#                 np.array([[0.09, - math.pi, self.stage.floor_height], [0.15, math.pi, 0.3]])
#             if self.stage.rigid_objects[rigid_object].__class__.__name__ == 'Cuboid':
#                 self.testing_intervention_spaces[rigid_object]['size'] = \
#                     np.array([[0.065, 0.065, 0.065], [0.075, 0.075, 0.075]])
# >>>>>>> origin/master
            self.testing_intervention_spaces[rigid_object]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            self.testing_intervention_spaces[rigid_object]['mass'] = \
                np.array([0.1, 0.2])
        for visual_object in self.stage._visual_objects:
            self.testing_intervention_spaces[visual_object] = dict()
            self.testing_intervention_spaces[visual_object]['position'] = \
                np.array([(self.stage._floor_inner_bounding_box[1] -
                           self.stage._floor_inner_bounding_box[0]) * 1 / 2 + \
                          self.stage._floor_inner_bounding_box[0],
                          self.stage._floor_inner_bounding_box[1]])
            self.testing_intervention_spaces[visual_object]['size'] = \
                np.array([[0.065, 0.065, 0.065], [0.075, 0.075, 0.075]])
# =======
#                 np.array([[0.09, - math.pi, self.stage.floor_height], [0.15, math.pi, 0.3]])
#             if self.stage.visual_objects[visual_object].__class__.__name__ == 'SCuboid':
#                 self.testing_intervention_spaces[visual_object]['size'] = \
#                     np.array([[0.065, 0.065, 0.065], [0.075, 0.075, 0.075]])
# >>>>>>> origin/master
            self.testing_intervention_spaces[visual_object]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self.testing_intervention_spaces['floor_color'] = \
            np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self.testing_intervention_spaces['stage_color'] = \
            np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self.testing_intervention_spaces['floor_friction'] = \
            np.array([0.6, 0.8])
        for link in self.robot.get_link_names():
            self.testing_intervention_spaces[link] = dict()
            self.testing_intervention_spaces[link]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
            self.testing_intervention_spaces[link]['mass'] = \
                np.array([0.6, 0.8])
        return

    def get_desired_goal(self):
        """

        :return:
        """
        desired_goal = []
        for visual_goal in self.stage._visual_objects:
            desired_goal.append(self.stage._visual_objects[visual_goal]
                                .get_bounding_box())
        return np.array(desired_goal)

    def get_achieved_goal(self):
        """

        :return:
        """
        achieved_goal = []
        for rigid_object in self.stage._rigid_objects:
            if self.stage._rigid_objects[rigid_object].is_not_fixed:
                achieved_goal.append(self.stage._rigid_objects
                                     [rigid_object].get_bounding_box())
        return np.array(achieved_goal)

    def _goal_distance(self, achieved_goal, desired_goal):
        """

        :param achieved_goal:
        :param desired_goal:
        :return:
        """
        # intersection areas / union of all visual_objects
        intersection_area = 0
        #TODO: under the assumption that the visual objects dont intersect
        #TODO: deal with structured data for silhouettes
        union_area = 0
        for desired_subgoal_bb in desired_goal:
            union_area += get_bounding_box_area(desired_subgoal_bb)
            for rigid_object_bb in achieved_goal:
                intersection_area += get_intersection(
                    desired_subgoal_bb, rigid_object_bb)
        if union_area > 0:
            sparse_reward = intersection_area / float(union_area)
        else:
            sparse_reward = 1
        return sparse_reward

    def _update_success(self, goal_distance):
        """

        :param goal_distance:
        :return:
        """
        preliminary_success = self._check_preliminary_success(goal_distance)
        if preliminary_success:
            self.task_solved = True
            self.time_steps_elapsed_since_success += 1
        else:
            self.task_solved = False
            self.time_steps_elapsed_since_success = 0
        return

    def _check_preliminary_success(self, goal_distance):
        """

        :param goal_distance:
        :return:
        """
        if goal_distance > 0.9:
            return True
        else:
            return False

    def get_reward(self):
        """

        :return:
        """
        self._current_desired_goal = self.get_desired_goal()
        self._current_achieved_goal = self.get_achieved_goal()
        self._current_goal_distance = self._goal_distance(desired_goal=self._current_desired_goal,
                                                          achieved_goal=self._current_achieved_goal)
        self._update_success(self._current_goal_distance)
        if not self.task_params['is_goal_distance_dense']:
            if self.is_done():
                goal_distance = 1
            else:
                goal_distance = -1
        if self.task_params['calculate_additional_dense_rewards']:
            dense_rewards, update_task_state_dict = \
                self._calculate_dense_rewards(achieved_goal=self._current_achieved_goal,
                                              desired_goal=self._current_desired_goal)
            reward = np.sum(np.array(dense_rewards) *
                            self.task_params["dense_reward_weights"]) \
                            + self._current_goal_distance * \
                     self.task_params["sparse_reward_weight"]
            self._update_task_state(update_task_state_dict)
        else:
            reward = self._current_goal_distance * self.task_params["sparse_reward_weight"]
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """

        :param achieved_goal:
        :param desired_goal:
        :param info:
        :return:
        """
        goal_distance = self._goal_distance(desired_goal=desired_goal,
                                            achieved_goal=achieved_goal)
        # TODO: this is to avoid computing those things twice for the info dict but should be refactored soon.
        self.fractional_reward = goal_distance
        self.desired_goal = desired_goal
        self.achieved_goal = achieved_goal

        if not self.task_params['is_goal_distance_dense']:
            #TODO: not exactly right, but its a limitation of HER
            if self._check_preliminary_success(goal_distance):
                goal_distance = 1
            else:
                goal_distance = -1
        reward = goal_distance * self.task_params["sparse_reward_weight"]
        return reward

    def init_task(self, robot, stage):
        """

        :param robot:
        :param stage:
        :return:
        """
        self.robot = robot
        self.stage = stage
        if self.task_params["joint_positions"] is not None:
            self.robot.reset_state(
                joint_positions=
                np.array(self.task_params["joint_positions"]),
                joint_velocities=np.zeros([9, ]))
        else:
            self.robot.reset_state(
                joint_positions=self.robot.get_rest_pose()[0],
                joint_velocities=np.zeros([9, ]))
        self._empty_stage = self._save_pybullet_state()
        self._set_up_stage_arena()
        self._default_starting_state = \
            self._save_state()
        self._current_starting_state = \
            self._save_state()
        self.stage.finalize_stage()
        self._set_up_non_default_observations()
        # self.task_params.update(self.initial_state)
        self._set_training_intervention_spaces()
        self._set_testing_intervention_spaces()
        self._set_task_state()
        return

    def _setup_non_default_robot_observation_key(self, observation_key,
                                                 observation_function,
                                                 lower_bound, upper_bound):
        """

        :param observation_key:
        :param observation_function:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        self.robot.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_robot_observation_funcs[observation_key] = \
            observation_function
        return

    def _setup_non_default_stage_observation_key(self, observation_key,
                                                 observation_function,
                                                 lower_bound, upper_bound):
        """

        :param observation_key:
        :param observation_function:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        self.stage.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_stage_observation_funcs[observation_key] = \
            observation_function
        return

    def reset_task(self, interventions_dict=None):
        """

        :param interventions_dict:
        :return:
        """
        # self.stage.clear_memory()
        # self.robot.tri_finger.reset_world()
        self.robot.clear()
        # self.stage.clear()
        #restore first to the starting state and
        # then potentially make the intervention]
        reset_observation_space_signal = self._restore_state(self._current_starting_state)
        # self._create_world()
        # self.stage.update_stage()
        self.task_solved = False
        self.finished_episode = False
        self.time_steps_elapsed_since_success = 0
        success_signal = None
        interventions_info = None
        if interventions_dict is not None:
            success_signal, interventions_info, reset_observation_space_signal = \
                self.apply_interventions(interventions_dict,
                                         check_bounds=
                                         self.task_params['intervention_split'])
            if success_signal:
                self._current_starting_state = self._save_state()
        self._set_task_state()
        return success_signal, interventions_info, reset_observation_space_signal

    def filter_structured_observations(self):
        """

        :return:
        """
        robot_observations_dict = self.robot.\
            get_current_observations(self._robot_observation_helper_keys)
        stage_observations_dict = self.stage.\
            get_current_observations(self._stage_observation_helper_keys)
        self.current_full_observations_dict = dict(robot_observations_dict)
        self.current_full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.task_robot_observation_keys:
            # dont forget to handle non standard observation here
            if key in self._non_default_robot_observation_funcs:
                if self.robot._normalize_observations:
                    normalized_observations = \
                        self.robot.\
                            normalize_observation_for_key\
                            (key=key, observation= self._non_default_robot_observation_funcs[key]())
                    observations_filtered = \
                        np.append(observations_filtered,
                                  normalized_observations)
                else:
                    observations_filtered =\
                        np.append(observations_filtered,
                                  self._non_default_robot_observation_funcs[key]())
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(self.current_full_observations_dict[key]))

        for key in self.task_stage_observation_keys:
            if key in self._non_default_stage_observation_funcs:
                if self.stage._normalize_observations:
                    normalized_observations = \
                        self.stage.normalize_observation_for_key\
                            (key=key,
                             observation=self._non_default_stage_observation_funcs[key]())
                    observations_filtered = \
                        np.append(observations_filtered,
                                  normalized_observations)
                else:
                    observations_filtered =\
                        np.append(observations_filtered,
                                  self._non_default_robot_observation_funcs[key]())
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(self.current_full_observations_dict[key]))

        return observations_filtered

    def get_task_params(self):
        """

        :return:
        """
        return self.task_params

    def is_done(self):
        """

        :return:
        """
        #here we consider that you succeeded if u stayed 0.1 sec in
        #the goal position
        if self.finished_episode:
            return True
        if self.task_params['time_threshold_in_goal_state_secs'] <= \
                (self.robot.get_dt() * self.time_steps_elapsed_since_success):
            self.finished_episode = True
# =======
#                 (self.robot.dt * self.time_steps_elapsed_since_success):
#             # self.finished_episode = True
#             # TODO: Disable terminal states for now
#             self.finished_episode = False
# >>>>>>> origin/master
        return self.finished_episode

    def set_sparse_reward(self, sparse_reward_weight):
        """

        :param sparse_reward_weight:
        :return:
        """
        self.task_params["sparse_reward_weight"] = \
            sparse_reward_weight
        return

    def do_single_random_intervention(self):
        """

        :return:
        """
        interventions_dict = dict()
        if self.task_params['training']:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        # choose random variable one intervention  only and intervene
        if len(intervention_space) == 0:
            return False, {}, {}
        variable_name = np.random.choice(list(intervention_space))
        variable_space = intervention_space[variable_name]
        sub_variable_name = None
        # if its a block then choose a property
        if isinstance(variable_space, dict):
            sub_variable_name = np.random.choice(list(variable_space.keys()))
            variable_space = variable_space[sub_variable_name]
        chosen_intervention = np.random.uniform(variable_space[0],
                                               variable_space[1])
        if sub_variable_name is None:
            interventions_dict[variable_name] = \
                chosen_intervention
        else:
            interventions_dict[variable_name] = dict()
            interventions_dict[variable_name][sub_variable_name] = \
                chosen_intervention
        success_signal, interventions_info, reset_observation_space_signal = \
            self.apply_interventions(interventions_dict, check_bounds=False)
        # self._set_task_state()
        return success_signal, interventions_info, interventions_dict, \
               reset_observation_space_signal

    def get_training_intervention_spaces(self):
        """

        :return:
        """
        return self.training_intervention_spaces

    def get_testing_intervention_spaces(self):
        """

        :return:
        """
        return self.testing_intervention_spaces

    def get_current_scm_values(self):
        """

        :return:
        """
        variable_params = dict()
        #get the robots ones
        variable_params.\
            update(self.robot.get_current_scm_values())
        #get the arena
        variable_params. \
            update(self.stage.get_current_scm_values())
        #get the task specific params now
        variable_params. \
            update(self.get_task_generator_variables_values())
        return variable_params

    def get_current_task_parameters(self):
        """

        :return:
        """
        #this is all the variables that are available and exposed
        current_variables_values = self.get_current_scm_values()
        #filter them only if intervention spaces split is enforced
        if self.task_params['intervention_split']:
            #choose intervention space
            if self.task_params['training']:
                intervention_space = self.training_intervention_spaces
            else:
                intervention_space = self.testing_intervention_spaces
            task_params_dict = dict()
            for variable_name in intervention_space:
                if isinstance(
                        intervention_space[variable_name], dict):
                    task_params_dict[variable_name] = dict()
                    for subvariable_name in intervention_space[variable_name]:
                        task_params_dict[variable_name][subvariable_name] = \
                            current_variables_values[variable_name][subvariable_name]
                else:
                    task_params_dict[variable_name] = current_variables_values[variable_name]
        else:
            task_params_dict = dict(current_variables_values)
        # you can add task specific ones after that
        return task_params_dict

    def is_intervention_in_bounds(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        if self.task_params['training']:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        for intervention in interventions_dict:
            if intervention in intervention_space:
                if not isinstance(interventions_dict[intervention], dict):
                    if ((intervention_space[intervention][0] >
                         interventions_dict[intervention]).any() or \
                         (intervention_space[intervention][1]
                             < interventions_dict[intervention]).any()):
                        return False
                else:
                    for sub_variable_name in interventions_dict[intervention]:
                        # TODO: not happy with this hack but there is probably no other workaround without
                        #  translating everything to polar coordinates (not optimal for observation space)
                        if sub_variable_name == 'position':
                            cyl_position = cart2cyl(interventions_dict[intervention][sub_variable_name])
                            if sub_variable_name in intervention_space[intervention] and \
                                    ((intervention_space[intervention][sub_variable_name][0] > cyl_position).any()
                                     or (intervention_space[intervention][sub_variable_name][1] < cyl_position).any()):
                                return False
                        elif sub_variable_name in intervention_space[intervention] and \
                            ((intervention_space[intervention]
                            [sub_variable_name][0] >
                            interventions_dict[intervention][sub_variable_name]).any() or \
                                (intervention_space[intervention]
                                 [sub_variable_name][1] <
                                 interventions_dict[intervention][sub_variable_name]).any()):
                            return False
        return True

    def divide_intervention_dict(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        #TODO: for now a heuristic for naming conventions
        robot_intervention_keys = \
            self.robot.get_current_scm_values().keys()
        stage_intervention_keys = \
            self.stage.get_current_scm_values().keys()
        task_generator_intervention_keys = \
            self.get_task_generator_variables_values().keys()
        robot_interventions_dict = dict()
        stage_interventions_dict = dict()
        task_generator_interventions_dict = dict()
        for intervention in interventions_dict:
            if intervention in robot_intervention_keys:
                robot_interventions_dict[intervention] = \
                    interventions_dict[intervention]
            elif intervention in stage_intervention_keys:
                stage_interventions_dict[intervention] = \
                    interventions_dict[intervention]
            elif intervention in task_generator_intervention_keys:
                task_generator_interventions_dict[intervention] = \
                    interventions_dict[intervention]
        return robot_interventions_dict, \
               stage_interventions_dict, \
               task_generator_interventions_dict

    def apply_interventions(self, interventions_dict,
                            check_bounds=False):
        """

        :param interventions_dict:
        :param check_bounds:
        :return:
        """
        interventions_info = {'out_bounds': False,
                              'robot_infeasible': None,
                              'stage_infeasible': None,
                              'task_generator_infeasible': None}

        stage_infeasible = False
        robot_infeasible = False
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict):
            interventions_info['out_bounds'] = True
            return False, interventions_info, False
        interventions_dict = \
            dict(self._handle_contradictory_interventions(interventions_dict))
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict):
            interventions_info['out_bounds'] = True
            return False, interventions_info, False
        #now divide the interventions
        robot_interventions_dict, stage_interventions_dict, \
        task_generator_interventions_dict = \
            self.divide_intervention_dict(interventions_dict)
        # current_stage_state = self.stage.get_full_state()
        # if self.robot.is_initialized():
        #     current_robot_state = self.robot.get_full_state()
        # else:
        #     current_robot_state = self.robot.get_default_state()
        current_state = self._save_state()
        self.robot.apply_interventions(robot_interventions_dict)
        self.stage.apply_interventions(stage_interventions_dict)
        task_generator_intervention_success_signal, reset_observation_space_signal = \
            self.apply_task_generator_interventions \
                (task_generator_interventions_dict)
        #TODO: this is a hack for now to not check feasibility when adding/removing objects since
        #The stage state is quite different afterwards and it will be hard to restore its exact state
        #we dont handle this

        if len(task_generator_interventions_dict) == 0:
            pre_contact_check_state = self._save_pybullet_state()
            self.robot.step_simulation()
            if not self.stage.check_feasiblity_of_stage():
                stage_infeasible = True
            if not self.robot.check_feasibility_of_robot_state():
                robot_infeasible = True
            if stage_infeasible or robot_infeasible:
                self._restore_state(current_state)
            else:
                self._restore_pybullet_state(pre_contact_check_state)
        interventions_info['robot_infeasible'] = \
            robot_infeasible
        interventions_info['stage_infeasible'] = \
            stage_infeasible
        interventions_info['task_generator_infeasible'] = \
            not task_generator_intervention_success_signal
        return not robot_infeasible and \
               not stage_infeasible and \
               task_generator_intervention_success_signal, \
               interventions_info, reset_observation_space_signal

    def do_intervention(self, interventions_dict, check_bounds=None):
        """

        :param interventions_dict:
        :param check_bounds:
        :return:
        """
        if check_bounds is None:
            check_bounds = self.task_params['intervention_split']
        success_signal, interventions_info, reset_observation_space_signal = \
            self.apply_interventions(interventions_dict,
                                     check_bounds=check_bounds)
        # self._set_task_state()
        return success_signal, interventions_info, \
               reset_observation_space_signal

    def get_max_episode_length(self):
        if self.task_params["task_name"] == 'reaching':
            episode_length = 5
        else:
            episode_length = len(self.stage.get_rigid_objects()) * 10
        return episode_length
