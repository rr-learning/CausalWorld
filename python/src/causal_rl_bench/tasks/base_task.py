import numpy as np
from causal_rl_bench.utils.state_utils import get_intersection


class BaseTask(object):
    def __init__(self, task_name):
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
        self.time_steps_elapsed_since_success = 0
        self.time_threshold_in_goal_state_secs = 0.5
        self.current_time_secs = 0
        self.training_intervention_spaces = None
        self.testing_intervention_spaces = None
        self.initial_state = dict()
        self.finished_episode = False
        self.enforce_intervention_spaces_split = False
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        # handle the contradictory intervention that changes each other (objects -> silhouettes)
        # and other way around sometimes
        return interventions_dict

    def get_task_generator_variables_values(self):
        return {}

    def _set_up_stage_arena(self):
        return

    def _set_up_non_default_observations(self):
        return

    def apply_task_generator_interventions(self, interventions_dict):
        return True

    def get_info(self):
        return {}

    def get_reward(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError()

    def _reset_task(self):
        raise NotImplementedError()

    def _set_training_intervention_spaces(self):
        raise NotImplementedError()

    def _set_testing_intervention_spaces(self):
        raise NotImplementedError()

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self._set_up_stage_arena()
        self.stage.finalize_stage()
        self._set_up_non_default_observations()
        self._set_training_intervention_spaces()
        self._set_testing_intervention_spaces()
        # for task_param_varaible in self.initial_state:
        #     if task_param_varaible not in self.training_intervention_spaces:
        #         self.training_intervention_spaces[task_param_varaible] = \
        #             np.array([self.initial_state[task_param_varaible],
        #                       self.initial_state[task_param_varaible]])
        #         self.testing_intervention_spaces[task_param_varaible] = \
        #             np.array([self.initial_state[task_param_varaible],
        #                       self.initial_state[task_param_varaible]])
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

    def _compute_sparse_reward(self, achieved_goal,
                               desired_goal, info,
                               redundant_calulcation=False):
        if not redundant_calulcation:
            self.current_time += self.robot.dt
        if self.task_name == "reaching":
            current_end_effector_positions = achieved_goal
            current_dist_to_goal = np.abs(desired_goal -
                                          current_end_effector_positions)
            current_dist_to_goal_mean = np.mean(current_dist_to_goal)
            if not redundant_calulcation:
                if current_dist_to_goal_mean < 0.01:
                    self.task_solved = True
                    self.time_steps_elapsed_since_success += 1
                else:
                    self.task_solved = False
                    # restart again
                    self.time_steps_elapsed_since_success = 0
            return current_dist_to_goal_mean
        else:
            # intersection areas / union of all visual_objects
            intersection_area = 0
            #TODO: under the assumption that the visual objects dont intersect
            #TODO: deal with structured data for silhouettes
            union_area = 0
            for visual_object_key in self.stage.visual_objects:
                visual_object = self.stage.get_object(visual_object_key)
                union_area += visual_object.get_area()
                for rigid_object_key in self.stage.rigid_objects:
                    rigid_object = self.stage.get_object(rigid_object_key)
                    if rigid_object.is_not_fixed:
                        intersection_area += get_intersection(
                            visual_object.get_bounding_box(),
                            rigid_object.get_bounding_box())
            sparse_reward = intersection_area / float(union_area)
            if sparse_reward > 0.9:
                self.task_solved = True
                self.time_steps_elapsed_since_success += 1
            else:
                self.task_solved = False
                # restart again
                self.time_steps_elapsed_since_success = 0
            return sparse_reward

    def enforce_intervention_spaces_split(self):
        self.enforce_intervention_spaces_split = True

    def reset_task(self, interventions_dict=None, is_training=True):
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
                                         is_training=is_training,
                                         check_bounds=
                                         self.enforce_intervention_spaces_split)
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
                                         is_training=is_training,
                                         check_bounds=False)

        else:
            self.apply_interventions(self.initial_state,
                                     is_training=is_training,
                                     check_bounds=False)
        self._reset_task()
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
                observations_filtered =\
                    np.append(observations_filtered,
                              self._non_default_robot_observation_funcs[key]())
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(self.current_full_observations_dict[key]))

        for key in self.task_stage_observation_keys:
            if key in self._non_default_stage_observation_funcs:
                observations_filtered = \
                    np.append(observations_filtered,
                              self._non_default_stage_observation_funcs[key]())
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

    def do_single_random_intervention(self, training_space=True):
        interventions_dict = dict()
        if training_space:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        # choose random variable one intervention  only and intervene
        variable_name = np.random.choice(list(intervention_space))
        variable_space = intervention_space[variable_name]
        sub_variable_name = None
        # if its a block then choose a property
        if isinstance(variable_space, dict):
            sub_variable_name = np.random.choice(list(variable_space.keys()))
            variable_space = variable_space[sub_variable_name]
        chosen_intervention = np.random.uniform(variable_space[0],
                                               variable_space[1])
        self.do_intervention(variable_name,
                             chosen_intervention,
                             sub_variable_name=sub_variable_name,
                             training=False)
        if isinstance(variable_space, dict):
            interventions_dict[variable_name] = dict()
            interventions_dict[variable_name][sub_variable_name] = \
                chosen_intervention
        else:
            interventions_dict[variable_name] = chosen_intervention
        return interventions_dict

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
        task_params_dict = dict()
        current_variables_values = self.get_current_variables_values()
        #filter them if they are not exposed in in the intervention spaces
        for variable_name in self.training_intervention_spaces:
            if isinstance(
                    self.training_intervention_spaces[variable_name], dict):
                task_params_dict[variable_name] = dict()
                for subvariable_name in self.training_intervention_spaces[variable_name]:
                    task_params_dict[variable_name][subvariable_name] = \
                        current_variables_values[variable_name][subvariable_name]
            else:
                task_params_dict[variable_name] = current_variables_values[variable_name]
        # you can add task specific ones after that
        return task_params_dict

    def is_intervention_in_bounds(self, interventions_dict,
                                  is_training=True):
        if is_training:
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
                            is_training=True,
                            check_bounds=False):
        interventions_info = {'out_bounds': False,
                              'robot_infeasible': None,
                              'stage_infeasible': None,
                              'task_generator_infeasible': None}
        if check_bounds and not self.is_intervention_in_bounds(
                interventions_dict, is_training=is_training):
            interventions_info['out_bounds'] = True
            return False, interventions_info
        interventions_dict = \
            dict(self._handle_contradictory_interventions(interventions_dict))
        if check_bounds and not self.is_intervention_in_bounds(interventions_dict,
                                                               is_training=is_training):
            interventions_info['out_bounds'] = True
            return False, interventions_info
        #now divide the interventions
        robot_interventions_dict, stage_interventions_dict, \
        task_generator_interventions_dict = \
            self.divide_intervention_dict(interventions_dict)
        robot_intervention_success_signal = \
            self.robot.apply_interventions(robot_interventions_dict)
        stage_intervention_success_signal = \
            self.stage.apply_interventions(stage_interventions_dict)
        task_generator_intervention_success_signal = \
            self.apply_task_generator_interventions\
                (task_generator_interventions_dict)
        interventions_info['robot_infeasible'] = not robot_intervention_success_signal
        interventions_info['stage_infeasible'] = not stage_intervention_success_signal
        interventions_info['task_generator_infeasible'] = not task_generator_intervention_success_signal
        return robot_intervention_success_signal and \
               stage_intervention_success_signal and \
               task_generator_intervention_success_signal, interventions_info





