import numpy as np
import math
from causal_rl_bench.utils.state_utils import get_bounding_box_area
from causal_rl_bench.utils.state_utils import get_intersection


class BaseTask(object):
    def __init__(self, task_name, intervention_split,
                 training, sparse_reward_weight=1,
                 dense_reward_weights=np.array([]),
                 is_goal_distance_dense=True,
                 calculate_additional_dense_rewards=True):
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
        self.task_params["sparse_reward_weight"] = sparse_reward_weight
        self.task_params["dense_reward_weights"] = dense_reward_weights
        self.time_steps_elapsed_since_success = 0
        self.time_threshold_in_goal_state_secs = 0.5
        self.current_time_secs = 0
        self.training_intervention_spaces = dict()
        self.testing_intervention_spaces = dict()
        self.initial_state = dict()
        self.finished_episode = False
        self.intervention_spaces_split = intervention_split
        self.training_intervention_mode = training
        self.is_goal_distance_dense = is_goal_distance_dense
        self.calculate_additional_dense_rewards = \
            calculate_additional_dense_rewards
        return

    def sample_new_goal(self):
        raise Exception("Sampling new goal is not supported for this "
                        "task")

    def get_description(self):
        return

    def _set_task_state(self):
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        # handle the contradictory intervention
        # that changes each other (objects -> silhouettes)
        # and other way around sometimes
        return interventions_dict

    def _set_up_stage_arena(self):
        return

    def _set_up_non_default_observations(self):
        return

    def get_task_generator_variables_values(self):
        return {}

    def apply_task_generator_interventions(self, interventions_dict):
        return True

    def get_info(self):
        return {}

    def _update_task_state(self, update_task_state_dict):
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        return np.array([]), None

    def _set_training_intervention_spaces(self):
        #you can override these easily
        self.training_intervention_spaces = dict()
        self.training_intervention_spaces['joint_positions'] = \
            np.array([[-math.radians(70), -math.radians(70),
                       -math.radians(160)] * 3,
                      [math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3])
        #any goal or object in arena put the position
        #and orientation modification
        for rigid_object in self.stage.rigid_objects:
            self.training_intervention_spaces[rigid_object] = dict()
            self.training_intervention_spaces[rigid_object]['position'] = \
                np.array([self.stage.floor_inner_bounding_box[0],
                          (self.stage.floor_inner_bounding_box[0] +
                           self.stage.floor_inner_bounding_box[1]) / 2])
        for visual_object in self.stage.visual_objects:
            self.training_intervention_spaces[visual_object] = dict()
            self.training_intervention_spaces[visual_object]['position'] = \
                np.array([self.stage.floor_inner_bounding_box[0],
                          (self.stage.floor_inner_bounding_box[0] +
                           self.stage.floor_inner_bounding_box[1]) / 2])
        return

    def _set_testing_intervention_spaces(self):
        # you can override these easily
        self.testing_intervention_spaces = dict()
        self.testing_intervention_spaces['joint_positions'] = \
            np.array([[math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3,
                      [math.radians(70), 0,
                       math.radians(-2)] * 3])
        # any goal or object in arena put the position
        # and orientation modification
        for rigid_object in self.stage.rigid_objects:
            self.testing_intervention_spaces[rigid_object] = dict()
            self.testing_intervention_spaces[rigid_object]['position'] = \
                np.array([(self.stage.floor_inner_bounding_box[0] +
                           self.stage.floor_inner_bounding_box[1]) / 2,
                          self.stage.floor_inner_bounding_box[1]])
        for visual_object in self.stage.visual_objects:
            self.testing_intervention_spaces[visual_object] = dict()
            self.testing_intervention_spaces[visual_object]['position'] = \
                np.array([(self.stage.floor_inner_bounding_box[0] +
                           self.stage.floor_inner_bounding_box[1]) / 2,
                          self.stage.floor_inner_bounding_box[1]])
        return

    def get_desired_goal(self):
        desired_goal = []
        for visual_goal in self.stage.visual_objects:
            desired_goal.append(self.stage.visual_objects[visual_goal]
                                .get_bounding_box())
        return np.array(desired_goal)

    def get_achieved_goal(self):
        achieved_goal = []
        for rigid_object in self.stage.rigid_objects:
            if self.stage.rigid_objects[rigid_object].is_not_fixed:
                achieved_goal.append(self.stage.rigid_objects
                                     [rigid_object].get_bounding_box())
        return np.array(achieved_goal)

    def _goal_distance(self, achieved_goal, desired_goal):
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
        preliminary_success = self._check_preliminary_success(goal_distance)
        if preliminary_success:
            self.task_solved = True
            self.time_steps_elapsed_since_success += 1
        else:
            self.task_solved = False
            self.time_steps_elapsed_since_success = 0
        return

    def _check_preliminary_success(self, goal_distance):
        if goal_distance > 0.9:
            return True
        else:
            return False

    def get_reward(self):
        desired_goal = self.get_desired_goal()
        achieved_goal = self.get_achieved_goal()
        goal_distance = self._goal_distance(desired_goal=desired_goal,
                                            achieved_goal=achieved_goal)
        self._update_success(goal_distance)
        if not self.is_goal_distance_dense:
            if self.is_done():
                goal_distance = 1
            else:
                goal_distance = -1
        if self.calculate_additional_dense_rewards:
            dense_rewards, update_task_state_dict = self._calculate_dense_rewards(achieved_goal=achieved_goal,
                                                                                  desired_goal=desired_goal)
            reward = np.sum(np.array(dense_rewards) *
                            self.task_params["dense_reward_weights"]) \
                            + goal_distance * self.task_params["sparse_reward_weight"]
            self._update_task_state(update_task_state_dict)
        else:
            reward = goal_distance * self.task_params["sparse_reward_weight"]
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        goal_distance = self._goal_distance(desired_goal=desired_goal,
                                            achieved_goal=achieved_goal)
        if not self.is_goal_distance_dense:
            #TODO: not exactly right, but its a limitation of HER
            if self._check_preliminary_success(goal_distance):
                goal_distance = 1
            else:
                goal_distance = -1
        reward = goal_distance * self.task_params["sparse_reward_weight"]
        return reward

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.initial_state['joint_positions'] = \
            self.robot.get_rest_pose()[0]
        self.initial_state['joint_velocities'] = \
            np.zeros([9, ])
        self._set_up_stage_arena()
        self.stage.finalize_stage()
        self._set_up_non_default_observations()
        self._set_training_intervention_spaces()
        self._set_testing_intervention_spaces()
        return

    def _setup_non_default_robot_observation_key(self, observation_key,
                                                 observation_function,
                                                 lower_bound, upper_bound):
        self.robot.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_robot_observation_funcs[observation_key] = \
            observation_function
        return

    def _setup_non_default_stage_observation_key(self, observation_key,
                                                 observation_function,
                                                 lower_bound, upper_bound):
        self.stage.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_stage_observation_funcs[observation_key] = \
            observation_function
        return

    def reset_task(self, interventions_dict=None):
        self.robot.clear()
        self.stage.clear()
        self.task_solved = False
        self.finished_episode = False
        self.time_steps_elapsed_since_success = 0
        self.current_time = 0
        success_signal = None
        interventions_info = None
        if interventions_dict is not None:
            interventions_dict_copy = interventions_dict
            non_changed_variables = \
                set(self.initial_state) - set(interventions_dict_copy)
            if len(non_changed_variables) > 0:
                interventions_dict_copy = dict(interventions_dict)
            for non_changed_variable in non_changed_variables:
                if isinstance(self.initial_state[non_changed_variable], dict):
                    for subvariable in self.initial_state[
                        non_changed_variable]:
                        interventions_dict_copy[non_changed_variable][
                            subvariable] = \
                            self.initial_state[non_changed_variable][
                                subvariable]
                else:
                    interventions_dict_copy[non_changed_variable] = \
                        self.initial_state[non_changed_variable]
            success_signal, interventions_info = \
                self.apply_interventions(interventions_dict_copy,
                                         check_bounds=
                                         self.intervention_spaces_split)
            if success_signal:
                for intervention_variable in self.initial_state:
                    if intervention_variable in interventions_dict:
                        if isinstance(self.initial_state[intervention_variable],dict):
                            for subvariable in self.initial_state[intervention_variable]:
                                if subvariable in interventions_dict[intervention_variable]:
                                    self.initial_state[intervention_variable] = \
                                        interventions_dict_copy[
                                            intervention_variable]
                        else:
                            self.initial_state[intervention_variable] = \
                                interventions_dict_copy[intervention_variable]
            else:
                self.apply_interventions(self.initial_state,
                                         check_bounds=False)

        else:
            self.apply_interventions(self.initial_state,
                                     check_bounds=False)
        self._set_task_state()
        return success_signal, interventions_info

    def filter_structured_observations(self):
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
                if self.robot.normalize_observations:
                    normalized_observations = \
                        self.robot.\
                            normalize_observation_for_key(key=key,
                                                          observation=
                                                          self._non_default_robot_observation_funcs[key]())
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
                if self.stage.normalize_observations:
                    normalized_observations = \
                        self.stage.\
                            normalize_observation_for_key(key=key,
                                                          observation=
                                                          self._non_default_stage_observation_funcs[key]())
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
        return self.task_params

    def is_done(self):
        #here we consider that you succeeded if u stayed 0.5 sec in
        #the goal position
        if self.finished_episode:
            return True
        if self.time_threshold_in_goal_state_secs <= \
                (self.robot.dt * self.time_steps_elapsed_since_success):
            self.finished_episode = True
        return self.finished_episode

    def set_sparse_reward(self, sparse_reward_weight):
        self.task_params["sparse_reward_weight"] = \
            sparse_reward_weight
        return

    def do_single_random_intervention(self):
        interventions_dict = dict()
        if self.training_intervention_mode:
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

        success_signal, interventions_info = \
            self.apply_interventions(interventions_dict, check_bounds=False)
        self._set_task_state()
        return success_signal, interventions_info, interventions_dict

    def get_training_intervention_spaces(self):
        return self.training_intervention_spaces

    def get_testing_intervention_spaces(self):
        return self.testing_intervention_spaces

    def get_current_variables_values(self):
        variable_params = dict()
        #get the robots ones
        variable_params.\
            update(self.robot.get_current_variables_values())
        #get the arena
        variable_params. \
            update(self.stage.get_current_variables_values())
        #get the task specific params now
        variable_params. \
            update(self.get_task_generator_variables_values())
        return variable_params

    def get_current_task_parameters(self):
        #this is all the variables that are available and exposed
        current_variables_values = self.get_current_variables_values()
        #filter them only if intervention spaces split is enforced
        if self.intervention_spaces_split:
            task_params_dict = dict()
            for variable_name in self.training_intervention_spaces:
                if isinstance(
                        self.training_intervention_spaces[variable_name], dict):
                    task_params_dict[variable_name] = dict()
                    for subvariable_name in self.training_intervention_spaces[variable_name]:
                        task_params_dict[variable_name][subvariable_name] = \
                            current_variables_values[variable_name][subvariable_name]
                else:
                    task_params_dict[variable_name] = current_variables_values[variable_name]
        else:
            task_params_dict = dict(current_variables_values)
        # you can add task specific ones after that
        return task_params_dict

    def is_intervention_in_bounds(self, interventions_dict):
        if self.training_intervention_mode:
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
                        if sub_variable_name in intervention_space[intervention] and \
                            ((intervention_space[intervention]
                            [sub_variable_name][0] >
                            interventions_dict[intervention][sub_variable_name]).any() or \
                                (intervention_space[intervention]
                                 [sub_variable_name][1] <
                                 interventions_dict[intervention][sub_variable_name]).any()):
                            return False
        return True

    def divide_intervention_dict(self, interventions_dict):
        #TODO: for now a heuristic for naming conventions
        robot_intervention_keys = self.robot.get_current_variables_values().keys()
        stage_intervention_keys = self.stage.get_current_variables_values().keys()
        task_generator_intervention_keys = self.get_task_generator_variables_values().keys()
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
        interventions_info = {'out_bounds': False,
                              'robot_infeasible': None,
                              'stage_infeasible': None,
                              'task_generator_infeasible': None}

        stage_infeasible = False
        robot_infeasible = False
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict):
            interventions_info['out_bounds'] = True
            return False, interventions_info
        interventions_dict = \
            dict(self._handle_contradictory_interventions(interventions_dict))
        if check_bounds and not self.is_intervention_in_bounds(interventions_dict):
            interventions_info['out_bounds'] = True
            return False, interventions_info
        #now divide the interventions
        robot_interventions_dict, stage_interventions_dict, \
        task_generator_interventions_dict = \
            self.divide_intervention_dict(interventions_dict)
        current_stage_state = self.stage.get_full_state()
        if self.robot.is_initialized():
            current_robot_state = self.robot.get_full_state()
        else:
            current_robot_state = self.robot.get_default_state()
        self.stage.apply_interventions(stage_interventions_dict)
        self.robot.apply_interventions(robot_interventions_dict)
        if not self.stage.check_feasiblity_of_stage():
            self.stage.set_full_state(current_stage_state)
            stage_infeasible = True
        if not self.robot.check_feasibility_of_robot_state():
            self.robot.set_full_state(current_robot_state)
            robot_infeasible = True
        task_generator_intervention_success_signal = \
            self.apply_task_generator_interventions\
                (task_generator_interventions_dict)
        interventions_info['robot_infeasible'] = \
            robot_infeasible
        interventions_info['stage_infeasible'] = \
            stage_infeasible
        interventions_info['task_generator_infeasible'] = \
            task_generator_intervention_success_signal
        return not robot_infeasible and \
               not stage_infeasible and \
               task_generator_intervention_success_signal, \
               interventions_info

    def do_intervention(self, interventions_dict, check_bounds=None):
        if check_bounds is None:
            check_bounds = self.intervention_spaces_split
        success_signal, interventions_info = \
            self.apply_interventions(interventions_dict,
                                     check_bounds=check_bounds)
        self._set_task_state()
        return success_signal, interventions_info




