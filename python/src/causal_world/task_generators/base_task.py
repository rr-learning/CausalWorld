import numpy as np
import math
import copy
from causal_world.utils.state_utils import get_bounding_box_volume
from causal_world.utils.state_utils import get_intersection
from causal_world.utils.rotation_utils import cart2cyl
from causal_world.utils.task_utils import combine_intervention_spaces
import pybullet
import logging


class BaseTask(object):

    def __init__(self,
                 task_name,
                 variables_space,
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 activate_sparse_reward=False):
        """
        This class represents the base task generator which includes all the
        common functionalities of the task generators.

        :param task_name: (str) the task name
        :param variables_space: (str) space to be used either 'space_a' or 'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the volumetric
                                              fraction overlap more than 90%.
        """
        self._robot = None
        self._stage = None
        self._task_solved = False
        self._task_name = task_name
        self._task_robot_observation_keys = []
        self._task_stage_observation_keys = []
        # the helper keys are observations that are not included in the task observations but it will be needed in reward
        # calculation or new observations calculation
        self._robot_observation_helper_keys = []
        self._stage_observation_helper_keys = []
        self._non_default_robot_observation_funcs = dict()
        self._non_default_stage_observation_funcs = dict()
        self._current_full_observations_dict = dict()
        self._task_params = dict()
        self._task_params["task_name"] = self._task_name
        self._task_params["fractional_reward_weight"] = fractional_reward_weight
        self._task_params["dense_reward_weights"] = dense_reward_weights
        self._task_params['activate_sparse_reward'] = activate_sparse_reward
        self._intervention_space_a = dict()
        self._intervention_space_b = dict()
        self._intervention_space_a_b = dict()
        self._task_params['variables_space'] = variables_space
        self._task_params["joint_positions"] = None
        self._current_starting_state = dict()
        self._default_starting_state = dict()
        self._empty_stage = None
        self._recreation_time = 0
        #TODO: this should be increased! there might be a bug from pybullet's side.
        self._period_to_clear_memory = 1
        self._current_desired_goal = None
        self._current_achieved_goal = None
        self._current_goal_reward = None
        self._max_episode_length = None
        self._create_world_func = None
        self._is_partial_solution_exposed = False
        self._is_ground_truth_state_exposed = False
        return

    def set_intervention_space(self, variables_space):
        """

        :param variables_space:

        :return:
        """
        self._task_params['variables_space'] = variables_space
        return

    def _adjust_variable_spaces_after_intervention(self, interventions_dict):
        return

    def _save_pybullet_state(self):
        """

        :return:
        """
        pybullet_state = dict()
        if self._stage._pybullet_client_full_id is not None:
            pybullet_state['full'] = pybullet.\
                saveState(physicsClientId=self._stage._pybullet_client_full_id)
        if self._stage._pybullet_client_w_goal_id is not None:
            pybullet_state['w_goal'] = pybullet.\
                saveState(physicsClientId=self._stage._pybullet_client_w_goal_id)
        if self._stage._pybullet_client_w_o_goal_id is not None:
            pybullet_state['w_o_goal'] = pybullet. \
                saveState(physicsClientId=self._stage._pybullet_client_w_o_goal_id)
        pybullet_state['latest_full_state'] = copy.deepcopy(
            self._robot.get_latest_full_state())
        return pybullet_state

    def _restore_pybullet_state(self, pybullet_state):
        """

        :param pybullet_state:

        :return:
        """
        if self._stage._pybullet_client_full_id is not None:
            pybullet.\
                restoreState(pybullet_state['full'],
                             physicsClientId=self._stage._pybullet_client_full_id)
        if self._stage._pybullet_client_w_goal_id is not None:
            pybullet.\
                restoreState(pybullet_state['w_goal'],
                             physicsClientId=self._stage._pybullet_client_w_goal_id)
        if self._stage._pybullet_client_w_o_goal_id is not None:
            pybullet. \
                restoreState(pybullet_state['w_o_goal'],
                             physicsClientId=self._stage._pybullet_client_w_o_goal_id)
        self._robot._latest_full_state = copy.deepcopy(
            pybullet_state['latest_full_state'])
        return

    def _remove_pybullet_state(self, pybullet_state):
        """

        :param pybullet_state:

        :return:
        """
        if self._stage._pybullet_client_full_id is not None:
            pybullet. \
                removeState(pybullet_state['full'],
                             physicsClientId=self._stage._pybullet_client_full_id)
        if self._stage._pybullet_client_w_goal_id is not None:
            pybullet. \
                removeState(pybullet_state['w_goal'],
                             physicsClientId=self._stage._pybullet_client_w_goal_id)
        if self._stage._pybullet_client_w_o_goal_id is not None:
            pybullet. \
                removeState(pybullet_state['w_o_goal'],
                             physicsClientId=self._stage._pybullet_client_w_o_goal_id)
        return

    def save_state(self):
        """

        :return:
        """
        state = dict()
        # state['pybullet_states'] = \
        #     self._save_pybullet_state()
        state['stage_object_state'] = \
            self._stage.get_full_env_state()
        state['robot_object_state'] = \
            self._robot.get_full_env_state()
        state['latest_full_state'] = copy.deepcopy(
            self._robot.get_latest_full_state())
        state['task_observations'] = \
            copy.deepcopy(self._task_stage_observation_keys)

        return state

    def restore_state(self, state_dict, avoid_reloading_urdf=False):
        """

        :param state_dict:
        :param avoid_reloading_urdf:

        :return:
        """
        old_number_of_rigid_objects = len(self._stage.get_rigid_objects())
        old_number_of_visual_objects = len(self._stage.get_visual_objects())
        reset_observation_space = False
        self._stage.remove_everything()
        if not avoid_reloading_urdf and \
                self._recreation_time != 0 and \
                self._recreation_time % self._period_to_clear_memory == 0:
            self._create_world_func()
            self._robot._disable_velocity_control()
            self._robot.set_full_env_state(state_dict['robot_object_state'])
            self._remove_pybullet_state(self._empty_stage)
            self._empty_stage = self._save_pybullet_state()
        else:
            self._restore_pybullet_state(self._empty_stage)
            self._robot.set_full_env_state(state_dict['robot_object_state'])
        self._stage.set_full_env_state(state_dict['stage_object_state'])
        self._recreation_time += 1
        new_number_of_rigid_objects = len(self._stage.get_rigid_objects())
        new_number_of_visual_objects = len(self._stage.get_visual_objects())
        if old_number_of_rigid_objects != new_number_of_rigid_objects:
            reset_observation_space = True
        if old_number_of_visual_objects != new_number_of_visual_objects:
            reset_observation_space = True
        self._task_stage_observation_keys = copy.deepcopy(
            state_dict['task_observations'])
        self._robot._latest_full_state = copy.deepcopy(
            state_dict['latest_full_state'])
        # self._restore_pybullet_state(state_dict['pybullet_states'])
        return reset_observation_space

    def get_variable_space_used(self):
        """

        :return:
        """
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_b
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
        return intervention_space

    def activate_sparse_reward(self):
        """

        :return:
        """
        self._task_params['activate_sparse_reward'] = True

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
        info['success'] = self._task_solved
        if self._is_ground_truth_state_exposed:
            info['ground_truth_current_state_varibales'] = \
                self.get_current_scm_values()
        if self._is_partial_solution_exposed:
            info['possible_solution_intervention'] = dict()
            for rigid_object in self._stage._rigid_objects:
                #check if there is an equivilant visual object corresponding
                possible_corresponding_goal = rigid_object.replace(
                    'tool', 'goal')
                if possible_corresponding_goal in self._stage.get_visual_objects(
                ):
                    info['possible_solution_intervention'][rigid_object] = dict(
                    )
                    info['possible_solution_intervention'][rigid_object]['cartesian_position'] = \
                        self._stage.get_object_state(possible_corresponding_goal, 'cartesian_position')
                    info['possible_solution_intervention'][rigid_object]['orientation'] = \
                        self._stage.get_object_state(possible_corresponding_goal, 'orientation')
        info['fractional_success'] = self._current_goal_reward
        return info

    def expose_potential_partial_solution(self):
        """
        Specified to add the potential partial solution to the info dict.

        :return:
        """
        self._is_partial_solution_exposed = True
        return

    def add_ground_truth_state_to_info(self):
        """
        Specified to add the full ground truth state to the info dict.

        :return:
        """
        self._is_ground_truth_state_exposed = True
        return

    def _update_task_state(self, update_task_state_dict):
        """

        :param update_task_state_dict:

        :return:
        """
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """
        Specified by the task generator itself.

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        return np.array([]), None

    def sample_new_goal(self, level=None):
        """

        :param level:

        :return:
        """
        #TODO: for now we just vary position as a new goal
        #Need to generalize this
        intervention_dict = dict()
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_b
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
        for visual_object in self._stage.get_visual_objects():
            if visual_object in intervention_space:
                intervention_dict[visual_object] = dict()
                if 'cylindrical_position' in intervention_space[visual_object]:
                    intervention_dict[visual_object]['cylindrical_position'] = \
                        cart2cyl(self._stage.random_position(
                            height_limits=intervention_space[visual_object]
                                          ['cylindrical_position'][:, 2],
                            radius_limits=intervention_space[visual_object]
                                          ['cylindrical_position'][:, 0],
                            angle_limits=intervention_space[visual_object]
                                         ['cylindrical_position'][:, 1]))
                if 'euler_orientation' in intervention_space[visual_object]:
                    intervention_dict[visual_object]['euler_orientation'] = \
                        np.random.uniform(intervention_space[visual_object]['euler_orientation'][0],
                                          intervention_space[visual_object]['euler_orientation'][1])
        return intervention_dict

    def reset_default_state(self):
        """

        :return:
        """
        self.restore_state(self._default_starting_state)
        self._task_solved = False
        self._set_task_state()
        self._current_starting_state = copy.deepcopy\
            (self._default_starting_state)
        return

    def _set_intervention_space_a(self):
        """

        :return:
        """
        #you can override these easily
        self._intervention_space_a = dict()
        self._intervention_space_a['joint_positions'] = \
            np.array([[-1.57, -1.2, -3.0] * 3,
                      [-0.69, 0,
                       0] * 3])
        #TODO: add the rest of the exposed variables here
        for rigid_object in self._stage.get_rigid_objects():
            self._intervention_space_a[rigid_object] = dict()
            if self._stage.get_rigid_objects(
            )[rigid_object].is_not_fixed():
                height = self._stage.get_object_state(rigid_object, 'size')[-1]
                self._intervention_space_a[rigid_object]['cylindrical_position'] = \
                    np.array([[0.0, - math.pi, height/2.0], [0.11, math.pi, 0.15]])
                self._intervention_space_a[rigid_object]['euler_orientation'] = \
                    np.array([[0, 0, -math.pi], [0, 0, math.pi]])
            if self._stage.get_rigid_objects(
            )[rigid_object].__class__.__name__ == 'Cuboid':
                self._intervention_space_a[rigid_object]['size'] = \
                    np.array([[0.055, 0.055, 0.055], [0.075, 0.075, 0.075]])
            self._intervention_space_a[rigid_object]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
            if self._stage.get_rigid_objects(
            )[rigid_object].is_not_fixed():
                self._intervention_space_a[rigid_object]['mass'] = \
                    np.array([0.015, 0.045])
        for visual_object in self._stage._visual_objects:
            self._intervention_space_a[visual_object] = dict()
            height = self._stage.get_object_state(visual_object, 'size')[-1]
            self._intervention_space_a[visual_object]['cylindrical_position'] = \
                np.array([[0.0, - math.pi, height/2.0], [0.11, math.pi, 0.15]])
            self._intervention_space_a[visual_object]['euler_orientation'] = \
                np.array([[0, 0, -math.pi], [0, 0, math.pi]])
            if self._stage.get_visual_objects(
            )[visual_object].__class__.__name__ == 'SCuboid':
                self._intervention_space_a[visual_object]['size'] = \
                    np.array([[0.055, 0.055, 0.055], [0.075, 0.075, 0.075]])
            self._intervention_space_a[visual_object]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self._intervention_space_a['floor_color'] = \
            np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self._intervention_space_a['stage_color'] = \
            np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self._intervention_space_a['floor_friction'] = \
            np.array([0.3, 0.8])
        self._intervention_space_a['stage_friction'] = \
            np.array([0.3, 0.8])
        for link in self._robot.get_link_names():
            self._intervention_space_a[link] = dict()
            self._intervention_space_a[link]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            self._intervention_space_a[link]['mass'] = \
                np.array([0.2, 0.6])
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        # you can override these easily
        self._intervention_space_b = dict()
        self._intervention_space_b['joint_positions'] = \
            np.array([[-0.69, 0,
                       0] * 3,
                      [1.0, 1.57, 3.0] * 3])
        # TODO: add the rest of the exposed variables here
        for rigid_object in self._stage.get_rigid_objects():
            self._intervention_space_b[rigid_object] = dict()
            if self._stage.get_rigid_objects(
            )[rigid_object].is_not_fixed():
                height = self._stage.get_object_state(rigid_object, 'size')[-1]
                self._intervention_space_b[rigid_object]['cylindrical_position'] = \
                    np.array([[0.11, - math.pi, height/2.0], [0.15, math.pi, 0.3]])
                self._intervention_space_b[rigid_object]['euler_orientation'] = \
                    np.array([[0, 0, -math.pi], [0, 0, math.pi]])
            if self._stage.get_rigid_objects(
            )[rigid_object].__class__.__name__ == 'Cuboid':
                self._intervention_space_b[rigid_object]['size'] = \
                    np.array([[0.075, 0.075, 0.075], [0.095, 0.095, 0.095]])
            self._intervention_space_b[rigid_object]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
            if self._stage.get_rigid_objects(
            )[rigid_object].is_not_fixed():
                self._intervention_space_b[rigid_object]['mass'] = \
                    np.array([0.1, 0.2])
        for visual_object in self._stage.get_visual_objects():
            self._intervention_space_b[visual_object] = dict()
            height = self._stage.get_object_state(visual_object, 'size')[-1]
            self._intervention_space_b[visual_object]['cylindrical_position'] = \
                np.array([[0.11, - math.pi, height/2.0], [0.15, math.pi, 0.3]])
            self._intervention_space_b[visual_object]['euler_orientation'] = \
                np.array([[0, 0, -math.pi], [0, 0, math.pi]])
            if self._stage.get_visual_objects(
            )[visual_object].__class__.__name__ == 'SCuboid':
                self._intervention_space_b[visual_object]['size'] = \
                    np.array([[0.075, 0.075, 0.075], [0.095, 0.095, 0.095]])
            self._intervention_space_b[visual_object]['color'] = \
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self._intervention_space_b['floor_color'] = \
            np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self._intervention_space_b['stage_color'] = \
            np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
        self._intervention_space_b['floor_friction'] = \
            np.array([0.6, 0.8])
        self._intervention_space_b['stage_friction'] = \
            np.array([0.6, 0.8])
        for link in self._robot.get_link_names():
            self._intervention_space_b[link] = dict()
            self._intervention_space_b[link]['color'] = \
                np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
            self._intervention_space_b[link]['mass'] = \
                np.array([0.045, 0.1])
        return

    def _set_intervention_space_a_b(self):
        """

        :return:
        """
        # you can override these easily
        #iterate over intervention space a and space b and concatenate them
        self._intervention_space_a_b = dict()
        for variable in self._intervention_space_a:
            if isinstance(self._intervention_space_a[variable], dict):
                self._intervention_space_a_b[variable] = dict()
                for subvariable_name in self._intervention_space_a[variable]:
                    self._intervention_space_a_b[variable][subvariable_name] = \
                        combine_intervention_spaces(self._intervention_space_a[variable][subvariable_name],
                                                    self._intervention_space_b[variable][subvariable_name])
            else:
                self._intervention_space_a_b[variable] = combine_intervention_spaces(
                    self._intervention_space_a[variable],
                    self._intervention_space_b[variable])
        return

    def get_desired_goal(self):
        """

        :return:
        """
        desired_goal = []
        for visual_goal in self._stage.get_visual_objects():
            desired_goal.append(self._stage.get_visual_objects()
                                [visual_goal].get_bounding_box())
        return np.array(desired_goal)

    def get_achieved_goal(self):
        """

        :return:
        """
        achieved_goal = []
        for rigid_object in self._stage.get_rigid_objects():
            if self._stage.get_rigid_objects()[rigid_object].is_not_fixed():
                achieved_goal.append(self._stage.get_rigid_objects()
                                     [rigid_object].get_bounding_box())
        return np.array(achieved_goal)

    def _goal_reward(self, achieved_goal, desired_goal):
        """
        :param achieved_goal:
        :param desired_goal:

        :return:
        """
        # intersection areas / union of all visual_objects
        #reshape the tensors if they are flattened with HER
        achieved_goal = np.reshape(achieved_goal, [-1, 2, 3])
        desired_goal = np.reshape(desired_goal, [-1, 2, 3])
        intersection_area = 0
        #TODO: under the assumption that the visual objects dont intersect
        #TODO: deal with structured data for silhouettes
        union_area = 0
        for desired_subgoal_bb in desired_goal:
            union_area += get_bounding_box_volume(desired_subgoal_bb)
            for rigid_object_bb in achieved_goal:
                intersection_area += get_intersection(desired_subgoal_bb,
                                                      rigid_object_bb)
        if union_area > 0:
            sparse_reward = intersection_area / float(union_area)
        else:
            sparse_reward = 1
        return sparse_reward

    def _update_success(self, goal_reward):
        """

        :param goal_reward:

        :return:
        """
        preliminary_success = self._check_preliminary_success(goal_reward)
        if preliminary_success:
            self._task_solved = True
        else:
            self._task_solved = False
        return

    def _check_preliminary_success(self, goal_reward):
        """

        :param goal_reward:

        :return:
        """
        if goal_reward > 0.9:
            return True
        else:
            return False

    def get_reward(self):
        """

        :return:
        """
        self._current_desired_goal = self.get_desired_goal()
        self._current_achieved_goal = self.get_achieved_goal()
        self._current_goal_reward = self._goal_reward(
            desired_goal=self._current_desired_goal,
            achieved_goal=self._current_achieved_goal)
        goal_reward = self._current_goal_reward
        self._update_success(self._current_goal_reward)
        if self._task_params['activate_sparse_reward']:
            if self._task_solved:
                goal_reward = 1
            else:
                goal_reward = 0
            return goal_reward
        else:
            dense_rewards, update_task_state_dict = \
                self._calculate_dense_rewards(achieved_goal=self._current_achieved_goal,
                                              desired_goal=self._current_desired_goal)
            reward = np.sum(np.array(dense_rewards) *
                            self._task_params["dense_reward_weights"]) \
                        + goal_reward * \
                        self._task_params["fractional_reward_weight"]
            self._update_task_state(update_task_state_dict)
            return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """

        :param achieved_goal:
        :param desired_goal:
        :param info:

        :return:
        """
        goal_reward = self._goal_reward(desired_goal=desired_goal,
                                            achieved_goal=achieved_goal)
        if self._task_params['activate_sparse_reward']:
            #TODO: not exactly right, but its a limitation of HER
            if self._check_preliminary_success(goal_reward):
                goal_reward = 1
            else:
                goal_reward = 0
            return goal_reward
        else:
            reward = goal_reward * self._task_params[
                "fractional_reward_weight"]
            return reward

    def init_task(self, robot, stage, max_episode_length, create_world_func):
        """

        :param robot:
        :param stage:

        :return:
        """
        self._create_world_func = create_world_func
        self._robot = robot
        self._stage = stage
        if self._task_params["joint_positions"] is not None:
            self._robot.reset_state(joint_positions=np.array(
                self._task_params["joint_positions"]),
                                    joint_velocities=np.zeros([
                                        9,
                                    ]))
        else:
            self._robot.reset_state(
                joint_positions=self._robot.get_rest_pose()[0],
                joint_velocities=np.zeros([
                    9,
                ]))
        self._empty_stage = self._save_pybullet_state()
        self._set_up_stage_arena()
        self._default_starting_state = \
            self.save_state()
        self._current_starting_state = \
            self.save_state()
        self._stage.finalize_stage()
        if max_episode_length is None:
            self._max_episode_length = self.get_default_max_episode_length()
        else:
            self._max_episode_length = max_episode_length
        self._setup_non_default_robot_observation_key(
            'time_left_for_task',
            self._calculate_time_left,
            lower_bound=np.array([0]),
            upper_bound=np.array([self._max_episode_length]))
        self._set_up_non_default_observations()
        # self.task_params.update(self.initial_state)
        self._set_intervention_space_a()
        self._set_intervention_space_b()
        self._set_intervention_space_a_b()
        self._set_task_state()
        return

    def _calculate_time_left(self):
        """

        :return:
        """
        current_control_index = self._robot.get_control_index()
        time_spent = (current_control_index + 1) * self._robot.get_dt()
        return self._max_episode_length - time_spent

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
        self._robot.add_observation(observation_key,
                                    lower_bound=lower_bound,
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
        self._stage.add_observation(observation_key,
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound)
        self._non_default_stage_observation_funcs[observation_key] = \
            observation_function
        return

    def reset_task(self, interventions_dict=None):
        """
        TODO: NOTE: With the current implementation, no contact points are saved
        when resetting task
        :param interventions_dict:

        :return:
        """
        self._robot.clear()
        reset_observation_space_signal = \
            self.restore_state(self._current_starting_state)

        self._task_solved = False
        success_signal = None
        interventions_info = None
        if interventions_dict is not None:
            success_signal, interventions_info, reset_observation_space_signal = \
                self.apply_interventions(interventions_dict,
                                         check_bounds=True)
            if success_signal:
                self._current_starting_state = self.save_state()
        self._set_task_state()
        return success_signal, interventions_info, reset_observation_space_signal

    def filter_structured_observations(self):
        """

        :return:
        """
        robot_observations_dict = self._robot.\
            get_current_observations(self._robot_observation_helper_keys)
        stage_observations_dict = self._stage.\
            get_current_observations(self._stage_observation_helper_keys)
        self._current_full_observations_dict = dict(robot_observations_dict)
        self._current_full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self._task_robot_observation_keys:
            # dont forget to handle non standard observation here
            if key in self._non_default_robot_observation_funcs:
                if self._robot._normalize_observations:
                    normalized_observations = \
                        self._robot.\
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
                              np.array(self._current_full_observations_dict[key]))

        for key in self._task_stage_observation_keys:
            if key in self._non_default_stage_observation_funcs:
                if self._stage._normalize_observations:
                    normalized_observations = \
                        self._stage.normalize_observation_for_key\
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
                              np.array(self._current_full_observations_dict[key]))

        return observations_filtered

    def get_task_params(self):
        """

        :return:
        """
        return self._task_params

    def is_done(self):
        """

        :return:
        """
        #here we consider that you succeeded if u stayed 0.1 sec in
        #the goal position
        return False

    def do_single_random_intervention(self):
        """

        :return:
        """
        interventions_dict = dict()
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
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

    def get_intervention_space_a(self):
        """

        :return:
        """
        return self._intervention_space_a

    def get_intervention_space_b(self):
        """

        :return:
        """
        return self._intervention_space_b

    def get_intervention_space_a_b(self):
        """

        :return:
        """
        return self._intervention_space_a_b

    def get_current_scm_values(self):
        """

        :return:
        """
        variable_params = dict()
        #get the robots ones
        variable_params.\
            update(self._robot.get_current_scm_values())
        #get the arena
        variable_params. \
            update(self._stage.get_current_scm_values())
        #get the task specific params now
        variable_params. \
            update(self.get_task_generator_variables_values())
        return variable_params

    def get_current_state_variables(self):
        #TODO: settle this one here
        """

        :return:
        """
        #this is all the variables that are availaavailableble and exposed
        current_variables_values = self.get_current_scm_values()
        # if self._task_params['use_train_space_only']:
        #     intervention_space = self._intervention_space_a
        #     state_variables_dict = dict()
        #     for variable_name in intervention_space:
        #         if isinstance(intervention_space[variable_name], dict):
        #             state_variables_dict[variable_name] = dict()
        #             for subvariable_name in intervention_space[variable_name]:
        #                 state_variables_dict[variable_name][subvariable_name] = \
        #                     current_variables_values[variable_name][subvariable_name]
        #         else:
        #             state_variables_dict[
        #                 variable_name] = current_variables_values[variable_name]
        # else:
        #
        state_variables_dict = dict(current_variables_values)
        # you can add task specific ones after that
        return state_variables_dict

    def is_intervention_in_bounds(self, interventions_dict):
        """
        :param interventions_dict:

        :return:
        """
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_b
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
        for intervention in interventions_dict:
            if intervention in intervention_space:
                if not isinstance(interventions_dict[intervention], dict):
                    if ((intervention_space[intervention][0] >
                         interventions_dict[intervention]).any() or \
                         (intervention_space[intervention][1]
                             < interventions_dict[intervention]).any()):
                        logging.warning("lower bound is: " + str(intervention_space[intervention][0]))
                        logging.warning("applied: " + str(interventions_dict[intervention]))
                        logging.warning("upper bound is:" + str(intervention_space[intervention][1]))
                        return False
                else:
                    for sub_variable_name in interventions_dict[intervention]:
                        if sub_variable_name in intervention_space[intervention] and \
                            ((intervention_space[intervention]
                            [sub_variable_name][0] >
                            interventions_dict[intervention][sub_variable_name]).any() or \
                                (intervention_space[intervention]
                                 [sub_variable_name][1] <
                                 interventions_dict[intervention][sub_variable_name]).any()):
                            logging.warning("lower bound is:" + str(intervention_space[intervention][sub_variable_name][0]))
                            logging.warning("applied:" + str(interventions_dict[intervention][sub_variable_name]))
                            logging.warning("upper bound is:" + str(intervention_space[intervention][sub_variable_name][1]))
                            return False
                        elif sub_variable_name not in intervention_space[intervention]:
                            logging.warning("{} not in intervention_space")
                            return False
            else:
                return False
        return True

    def divide_intervention_dict(self, interventions_dict):
        """

        :param interventions_dict:

        :return:
        """
        #TODO: for now a heuristic for naming conventions
        robot_intervention_keys = \
            self._robot.get_current_scm_values().keys()
        stage_intervention_keys = \
            self._stage.get_current_scm_values().keys()
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

    def apply_interventions(self, interventions_dict, check_bounds=False):
        """

        :param interventions_dict:
        :param check_bounds:

        :return:
        """
        interventions_info = {
            'out_bounds': False,
            'robot_infeasible': None,
            'stage_infeasible': None,
            'task_generator_infeasible': None
        }

        stage_infeasible = False
        robot_infeasible = False
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict):
            interventions_info['out_bounds'] = True
            logging.warning("Applying intervention is out of bounds " + str(interventions_dict))
            return False, interventions_info, False
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict):
            interventions_info['out_bounds'] = True
            logging.warning("Applying intervention is out of bounds " + str(interventions_dict))
            return False, interventions_info, False
        interventions_dict = \
            dict(self._handle_contradictory_interventions(interventions_dict))
        robot_interventions_dict, stage_interventions_dict, \
        task_generator_interventions_dict = \
            self.divide_intervention_dict(interventions_dict)
        current_state = self.save_state()
        self._robot.apply_interventions(robot_interventions_dict)
        self._stage.apply_interventions(stage_interventions_dict)
        task_generator_intervention_success_signal, reset_observation_space_signal = \
            self.apply_task_generator_interventions \
                (task_generator_interventions_dict)
        #TODO: this is a hack for now to not check feasibility when adding/removing objects since
        #The stage state is quite different afterwards and it will be hard to restore its exact state
        #we dont handle this
        if len(task_generator_interventions_dict) == 0:
            pre_contact_check_state = self._save_pybullet_state()
            self._robot.step_simulation()
            if not self._stage.check_feasiblity_of_stage():
                stage_infeasible = True
                logging.warning("Applying intervention lead to infeasibility of the stage")
            if not self._robot.check_feasibility_of_robot_state():
                robot_infeasible = True
                logging.warning("Applying intervention lead to infeasibility of the robot")
            if stage_infeasible or robot_infeasible:
                self.restore_state(current_state, avoid_reloading_urdf=True)
            else:
                self._restore_pybullet_state(pre_contact_check_state)
            self._remove_pybullet_state(pre_contact_check_state)
        interventions_info['robot_infeasible'] = \
            robot_infeasible
        interventions_info['stage_infeasible'] = \
            stage_infeasible
        interventions_info['task_generator_infeasible'] = \
            not task_generator_intervention_success_signal
        success = not robot_infeasible and \
               not stage_infeasible and \
               task_generator_intervention_success_signal
        if success:
            self._adjust_variable_spaces_after_intervention(interventions_dict)
        return success, \
               interventions_info, reset_observation_space_signal

    def do_intervention(self, interventions_dict, check_bounds=True):
        """

        :param interventions_dict:
        :param check_bounds:

        :return:
        """
        success_signal, interventions_info, reset_observation_space_signal = \
            self.apply_interventions(interventions_dict,
                                     check_bounds=check_bounds)
        # self._set_task_state()
        return success_signal, interventions_info, \
               reset_observation_space_signal

    def get_default_max_episode_length(self):
        """

        :return:
        """
        if self._task_params["task_name"] == 'reaching':
            episode_length = 5
        else:
            episode_length = len(self._stage.get_rigid_objects()) * 10
        return episode_length

    def get_task_name(self):
        """

        :return:
        """
        return self._task_name
