import pickle


class TaskStats:

    def __init__(self, task):
        """

        :param task:
        """
        self.task_name = task._task_name
        self.task_params = task.get_task_params()
        self.time_steps = 0
        self.num_resets = 0

    def add_episode_experience(self, time_steps):
        """

        :param time_steps:
        :return:
        """
        self.time_steps += time_steps
        self.num_resets += 1


class Tracker:

    def __init__(self, task=None, file_path=None, world_params=None):
        """

        :param task:
        :param file_path:
        :param world_params:
        """
        self.total_time_steps = 0
        self.total_resets = 0
        self.total_interventions = 0
        self.total_intervention_steps = 0
        self.invalid_intervention_steps = 0
        self.invalid_out_of_bounds_intervention_steps = 0
        self.invalid_robot_intervention_steps = 0
        self.invalid_stage_intervention_steps = 0
        self.invalid_task_generator_intervention_steps = 0
        self.task_stats_log = []
        if task is None:
            self._curr_task_stat = None
        else:
            self._curr_task_stat = TaskStats(task)
        if file_path is not None:
            self.load(file_path)
            if world_params is not None:
                if self.world_params != world_params:
                    raise Exception("Incompatible world params")
        else:
            self.world_params = world_params

    def add_episode_experience(self, time_steps):
        """

        :param time_steps:

        :return:
        """
        if self._curr_task_stat is None:
            raise Exception("No current task stat set")
        if time_steps > 0:
            self._curr_task_stat.add_episode_experience(time_steps)

    def switch_task(self, task):
        """

        :param task:

        :return:
        """
        self.total_time_steps += self._curr_task_stat.time_steps
        self.total_resets += self._curr_task_stat.num_resets
        # Need to further discuss this but for now regard switching a task as intervention
        self.total_intervention_steps += 1
        self.task_stats_log.append(self._curr_task_stat)
        self._curr_task_stat = TaskStats(task)

    def do_intervention(self, task, interventions_dict):
        """

        :param task:
        :param interventions_dict:

        :return:
        """
        self.total_time_steps += self._curr_task_stat.time_steps
        self.total_resets += self._curr_task_stat.num_resets
        self.total_intervention_steps += 1
        self.task_stats_log.append(self._curr_task_stat)
        self._curr_task_stat = TaskStats(task)
        self.total_interventions += len(interventions_dict)

    def add_invalid_intervention(self, interventions_info):
        """

        :param interventions_info:

        :return:
        """
        self.invalid_intervention_steps += 1
        if interventions_info['robot_infeasible']:
            self.invalid_robot_intervention_steps += 1
        if interventions_info['stage_infeasible']:
            self.invalid_stage_intervention_steps += 1
        if interventions_info['task_generator_infeasible']:
            self.invalid_task_generator_intervention_steps += 1
        if interventions_info['out_bounds']:
            self.invalid_out_of_bounds_intervention_steps += 1
        return

    def save(self, file_path):
        """

        :param file_path:

        :return:
        """
        if self.world_params is None:
            raise Exception("world_params not set")
        tracker_dict = {
            "task_stats_log":
                self.task_stats_log + [self._curr_task_stat],
            "total_time_steps":
                self.total_time_steps + self._curr_task_stat.time_steps,
            "total_resets":
                self.total_resets + self._curr_task_stat.num_resets,
            "total_interventions":
                self.total_interventions,
            "total_intervention_steps":
                self.total_intervention_steps,
            "total_invalid_intervention_steps":
                self.invalid_intervention_steps,
            "total_invalid_robot_intervention_steps":
                self.invalid_robot_intervention_steps,
            "total_invalid_stage_intervention_steps":
                self.invalid_stage_intervention_steps,
            "total_invalid_task_generator_intervention_steps":
                self.invalid_task_generator_intervention_steps,
            "total_invalid_out_of_bounds_intervention_steps":
                self.invalid_out_of_bounds_intervention_steps,
            "world_params":
                self.world_params
        }
        with open(file_path, "wb") as file_handle:
            pickle.dump(tracker_dict, file_handle)

    def load(self, file_path):
        """

        :param file_path:

        :return:
        """
        with open(file_path, "rb") as file:
            tracker_dict = pickle.load(file)
            self.total_interventions += tracker_dict["total_interventions"]
            self.total_intervention_steps += tracker_dict[
                "total_intervention_steps"]
            self.invalid_intervention_steps += tracker_dict[
                "total_invalid_intervention_steps"]
            self.invalid_robot_intervention_steps += tracker_dict[
                "total_invalid_robot_intervention_steps"]
            self.invalid_stage_intervention_steps += tracker_dict[
                "total_invalid_stage_intervention_steps"]
            self.invalid_task_generator_intervention_steps += tracker_dict[
                "total_invalid_task_generator_intervention_steps"]
            self.invalid_out_of_bounds_intervention_steps += \
                tracker_dict["total_invalid_out_of_bounds_intervention_steps"]
            self.total_time_steps += tracker_dict["total_time_steps"]
            self.total_resets += tracker_dict["total_resets"]
            self.task_stats_log = tracker_dict["task_stats_log"]
            self.world_params = tracker_dict["world_params"]

    def get_total_intervention_steps(self):
        """

        :return:
        """
        return self.total_intervention_steps

    def get_total_interventions(self):
        """

        :return:
        """
        return self.total_interventions

    def get_total_resets(self):
        """

        :return:
        """
        return self.total_resets

    def get_total_time_steps(self):
        """

        :return:
        """
        return self.total_time_steps

    def get_total_invalid_intervention_steps(self):
        """

        :return:
        """
        return self.invalid_intervention_steps

    def get_total_invalid_robot_intervention_steps(self):
        """

        :return:
        """
        return self.invalid_robot_intervention_steps

    def get_total_invalid_stage_intervention_steps(self):
        """

        :return:
        """
        return self.invalid_stage_intervention_steps

    def get_total_invalid_task_generator_intervention_steps(self):
        """

        :return:
        """
        return self.invalid_task_generator_intervention_steps

    def get_total_invalid_out_of_bounds_intervention_steps(self):
        """

        :return:
        """
        return self.invalid_out_of_bounds_intervention_steps
