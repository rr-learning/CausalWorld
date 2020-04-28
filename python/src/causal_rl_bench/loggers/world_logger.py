from causal_rl_bench.loggers.episode import Episode

import pickle


class WorldLogger:
    """
    This class logs the full histories of a world across multiple episodes
    """

    def __init__(self):
        self.episodes = []
        self._curr = None

    def new_episode(self):
        if self._curr:
            # convert to dict for saving so loading has no dependencies
            self.episodes.append(self._curr.__dict__)

        self._curr = Episode()

    def append(self, robot_action, world_state, reward, timestamp):
        self._curr.append(robot_action, world_state, reward, timestamp)

    def store(self, filename):
        with open(filename, "wb") as file_handle:
            pickle.dump(self.episodes, file_handle)
