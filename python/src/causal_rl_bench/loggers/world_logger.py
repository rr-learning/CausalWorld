from causal_rl_bench.loggers.episode import Episode

import json
import pickle
import os


class WorldLogger:
    """
    This class logs the full histories of a world across multiple episodes
    """

    def __init__(self, filename, output_path=None, saving_frequency=10):
        self.saving_frequency = saving_frequency
        if output_path is None:
            self.path = os.path.join("output", "logs")
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        else:
            self.path = output_path
        self.file_path = os.path.join(self.path, filename)
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as file:
                self.episodes = pickle.load(file)
        else:
            self.episodes = []
        self._curr = None

    def new_episode(self, initial_full_state, task_params=None):
        if self._curr:
            self.episodes.append(self._curr)
        self._curr = Episode(task_params, initial_full_state)
        # if len(self.episodes) % self.saving_frequency == 0 and len(self.episodes) != 0:
        #     self.save()

    def append(self, robot_action, observation, reward, timestamp):
        self._curr.append(robot_action, observation, reward, timestamp)

    def save(self):
        if len(self._curr.observations):
            self.episodes.append(self._curr)

        with open(self.file_path, "wb") as file_handle:
            pickle.dump(self.episodes, file_handle)
            # for episode in self.episodes:
            #     json.dump(episode.__dict__, file_handle)
            # self.episodes = []

    def get_episode(self, number=0):
        if number >= len(self.episodes):
            raise Exception("World logger contains less episodes")
        return self.episodes[number]

    def get_number_of_logged_episodes(self):
        return len(self.episodes)
