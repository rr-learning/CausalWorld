from causal_rl_bench.loggers.episode import Episode

import pickle
import os


class DataRecorder:
    """
    This class logs the full histories of a world across multiple episodes
    """

    def __init__(self, output_path=None, rec_dumb_frequency=100):
        self.rec_dumb_frequency = rec_dumb_frequency
        if output_path is None:
            self.path = os.path.join("output", "logs")
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        self.episodes = []
        self.last_episode_number_dumbed = len(self.episodes)
        self._curr = None

    def new_episode(self, initial_full_state, task_params=None):
        if self._curr:
            self.episodes.append(self._curr)
        self._curr = Episode(task_params, initial_full_state)
        if len(self.episodes) % self.rec_dumb_frequency == 0 and len(self.episodes) != 0:
            self.save()

    def append(self, robot_action, observation, reward, timestamp):
        self._curr.append(robot_action, observation, reward, timestamp)

    def save(self):
        if len(self._curr.observations):
            self.episodes.append(self._curr)
        new_episode_number_dumbed = self.last_episode_number_dumbed + len(self.episodes)
        file_path = os.path.join(self.path, "episode_{}_{}".format(self.last_episode_number_dumbed,
                                                                   new_episode_number_dumbed))
        with open(file_path, "wb") as file_handle:
            pickle.dump(self.episodes, file_handle)
            self.last_episode_number_dumbed = new_episode_number_dumbed
            self.episodes = []

    def get_episode(self, number=0):
        if number >= len(self.episodes):
            raise Exception("World logger contains less episodes")
        return self.episodes[number]

    def get_number_of_logged_episodes(self):
        return len(self.episodes)
