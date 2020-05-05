from causal_rl_bench.loggers.episode import Episode

import pickle
import os


class DataLoader:
    def __init__(self, data_path=None):
        if data_path:
            if os.path.exists(data_path):
                with open(data_path, "rb") as file:
                    self.episodes = pickle.load(file)
            else:
                raise ValueError("data_path does not exist")
        else:
            self.episodes = None

    def load_data(self, data_path):
        if os.path.exists(data_path):
            with open(data_path, "rb") as file:
                self.episodes = pickle.load(file)
        else:
            raise ValueError("data_path does not exist")

    def get_episodes(self):
        return self.episodes
