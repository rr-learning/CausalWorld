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
        return

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self._set_up_stage_arena()
        self.stage.finalize_stage()
        self._set_up_non_default_observations()
        self._set_training_intervention_spaces()
        self._set_testing_intervention_spaces()
        return

    def _set_up_stage_arena(self):
        return

    def _set_up_non_default_observations(self):
        return

    def _setup_non_default_robot_observation_key(self, observation_key,
                                                 observation_function,
                                                 lower_bound, upper_bound):
        # observation function takes in full observations dict and returns a numpy array
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

    def _compute_sparse_reward(self, achieved_goal, desired_goal, info):
        self.current_time += self.robot.dt
        if self.task_name == "reaching":
            current_end_effector_positions = \
                self.robot.compute_end_effector_positions(
                    self.robot.latest_full_state.position)
            current_dist_to_goal = np.abs(desired_goal -
                                          current_end_effector_positions)
            current_dist_to_goal_mean = np.mean(current_dist_to_goal)
            if current_dist_to_goal_mean < 0.01:
                self.task_solved = True
                self.time_steps_elapsed_since_success += 1
                return 1
            else:
                self.task_solved = False
                #restart again
                self.time_steps_elapsed_since_success = 0
                return 0
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
                return 1
            else:
                self.task_solved = False
                # restart again
                self.time_steps_elapsed_since_success = 0
                return 0

    def reset_task(self, interventions_dict=None):
        self.robot.clear()
        self.stage.clear()
        self.task_solved = False
        self.finished_episode = False
        self.time_steps_elapsed_since_success = 0
        self.current_time = 0
        self._reset_task(interventions_dict)
        return

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

    def get_info(self):
        return {}

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

    def do_random_intervention(self, training_space=True):
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
            interventions_dict[variable_name][sub_variable_name] = chosen_intervention
        else:
            interventions_dict[variable_name] = chosen_intervention
        return interventions_dict

    def apply_interventions(self, interventions_dict, initial_state_latch=True):
        interventions_dict_copy = interventions_dict
        non_changed_variables = \
            set(self.initial_state) - set(interventions_dict_copy)
        if len(non_changed_variables) > 0:
            interventions_dict_copy = dict(interventions_dict)
        for non_changed_variable in non_changed_variables:
            interventions_dict_copy[non_changed_variable] = \
                self.initial_state[non_changed_variable]
        for intervention_key, intervention_value in interventions_dict_copy.items():
            # if its a block then choose a property
            if isinstance(intervention_value, dict):
                for sub_intervention_key, sub_intervention_value in \
                        intervention_value.items():
                    #TODO: take care about it later
                    self.do_intervention(intervention_key,
                                         sub_intervention_value,
                                         sub_variable_name=sub_intervention_key,
                                         training=False)
                    if intervention_key in self.initial_state.keys() and \
                            sub_intervention_key in \
                            self.initial_state[intervention_key].keys() and \
                            initial_state_latch:
                        self.initial_state[intervention_key][sub_intervention_key] \
                            = intervention_value
            else:
                sub_variable_name = None
                self.do_intervention(intervention_key,
                                     intervention_value,
                                     sub_variable_name=sub_variable_name,
                                     training=False)
                if intervention_key in self.initial_state.keys() and \
                        initial_state_latch:
                    self.initial_state[intervention_key] \
                        = intervention_value
        return

    def do_intervention(self, variable_name, variable_value,
                        sub_variable_name=None, training=True):
        #TODO: this now only supports two levels of variables
        raise NotImplementedError()

    def get_training_intervention_spaces(self):
        return self.training_intervention_spaces

    def get_testing_intervention_spaces(self):
        return self.testing_intervention_spaces

    def get_reward(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError()

    def _reset_task(self, interventions_dict):
        raise NotImplementedError()

    def _set_training_intervention_spaces(self):
        raise NotImplementedError()

    def _set_testing_intervention_spaces(self):
        raise NotImplementedError()

