import pickle


class TaskStats:
    def __init__(self, task):
        self.task_name = task.task_name
        self.task_params = task.get_task_params()
        self.time_steps = 0
        self.num_resets = 0

    def add_episode_experience(self, time_steps):
        self.time_steps += time_steps
        self.num_resets += 1


class Tracker:
    def __init__(self, task=None, file_path=None, world_params=None):
        self.total_time_steps = 0
        self.total_resets = 0
        self.total_interventions = 0
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
        if self._curr_task_stat is None:
            raise Exception("No current task stat set")
        if time_steps > 0:
            self._curr_task_stat.add_episode_experience(time_steps)

    def switch_task(self, task):
        self.total_time_steps += self._curr_task_stat.time_steps
        self.total_resets += self._curr_task_stat.num_resets
        # Need to further discuss this but for now regard switching a task as intervention
        self.total_interventions += 1
        self.task_stats_log.append(self._curr_task_stat)
        self._curr_task_stat = TaskStats(task)

    def do_intervention(self, task):
        self.total_time_steps += self._curr_task_stat.time_steps
        self.total_resets += self._curr_task_stat.num_resets
        self.total_interventions += 1
        self.task_stats_log.append(self._curr_task_stat)
        self._curr_task_stat = TaskStats(task)

    def save(self, file_path):
        if self.world_params is None:
            raise Exception("world_params not set")
        tracker_dict = {"task_stats_log": self.task_stats_log + [self._curr_task_stat],
                        "total_time_steps": self.total_time_steps + self._curr_task_stat.time_steps,
                        "total_resets": self.total_resets + self._curr_task_stat.num_resets,
                        "total_interventions": self.total_interventions,
                        "world_params": self.world_params}
        with open(file_path, "wb") as file_handle:
            pickle.dump(tracker_dict, file_handle)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            tracker_dict = pickle.load(file)
            self.total_interventions += tracker_dict["total_interventions"]
            self.total_time_steps += tracker_dict["total_time_steps"]
            self.total_resets += tracker_dict["total_resets"]
            self.task_stats_log = tracker_dict["task_stats_log"]
            self.world_params = tracker_dict["world_params"]


