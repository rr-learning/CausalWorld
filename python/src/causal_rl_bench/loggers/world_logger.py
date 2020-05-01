from causal_rl_bench.loggers.episode import Episode

import pickle


class WorldLogger:
    """
    This class logs the full histories of a world across multiple episodes
    """

    def __init__(self, filename=None):
        if filename is None:
            self.episodes = []
        else:
            with open(filename, "r") as file:
                self.episodes = pickle.load(file)
        self._curr = None

    def new_episode(self, task_params=None):
        if self._curr:
            # convert to dict for saving so loading has no dependencies
            self.episodes.append(self._curr.__dict__)
        self._curr = Episode(task_params)

    def append(self, robot_action, world_state, reward, timestamp):
        self._curr.append(robot_action, world_state, reward, timestamp)

    def save(self, filename):
        with open(filename, "wb") as file_handle:
            pickle.dump(self.episodes, file_handle)

    def get_episode(self, number=0):
        if number >= len(self.episodes):
            raise Exception("World logger contains less episodes")
        return self.episodes[number]

    def get_number_of_logged_episodes(self):
        return len(self.episodes)
