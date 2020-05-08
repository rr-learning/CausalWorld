import numpy as np


class BaseTask(object):
    def __init__(self, id):
        self.robot = None
        self.stage = None
        self.task_solved = False
        self.id = id
        self.task_robot_observation_keys = []
        self.task_stage_observation_keys = []
        # the helper keys are observations that are not included in the task observations but it will be needed in reward
        # calculation or new observations calculation
        self._robot_observation_helper_keys = []
        self._stage_observation_helper_keys = []
        self._non_default_robot_observation_funcs = dict()
        self._non_default_stage_observation_funcs = dict()
        self.current_full_observations_dict = dict()
        return

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self._set_up_stage_arena()
        self.stage.finalize_stage()
        self._set_up_non_default_observations()
        return

    def _set_up_stage_arena(self):
        return

    def _set_up_non_default_observations(self):
        return

    def _setup_non_default_robot_observation_key(self, observation_key, observation_function, lower_bound, upper_bound):
        # observation function takes in full observations dict and returns a numpy array
        self.robot.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_robot_observation_funcs[observation_key] = observation_function
        return

    def _setup_non_default_stage_observation_key(self, observation_key, observation_function, lower_bound, upper_bound):
        self.stage.add_observation(observation_key, lower_bound=lower_bound,
                                   upper_bound=upper_bound)
        self._non_default_stage_observation_funcs[observation_key] = observation_function
        return

    def reset_task(self):
        self.robot.clear()
        self.stage.clear()
        self.task_solved = False
        self._reset_task()
        return

    def filter_structured_observations(self):
        robot_observations_dict = self.robot.get_current_observations(self._robot_observation_helper_keys)
        stage_observations_dict = self.stage.get_current_observations(self._stage_observation_helper_keys)
        self.current_full_observations_dict = dict(robot_observations_dict)
        self.current_full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.task_robot_observation_keys:
            # dont forget to handle non standard observation here
            if key in self._non_default_robot_observation_funcs:
                observations_filtered = np.append(observations_filtered,
                                                  self._non_default_robot_observation_funcs[key]())
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(self.current_full_observations_dict[key]))

        for key in self.task_stage_observation_keys:
            if key in self._non_default_stage_observation_funcs:
                observations_filtered = np.append(observations_filtered,
                                                  self._non_default_stage_observation_funcs[key]())
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(self.current_full_observations_dict[key]))

        return observations_filtered

    def get_task_params(self):
        raise NotImplementedError()

    def _reset_task(self):
        raise NotImplementedError()

    def is_done(self):
        raise NotImplementedError()

    def do_random_intervention(self):
        raise NotImplementedError()

    def do_intervention(self, **kwargs):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError()

