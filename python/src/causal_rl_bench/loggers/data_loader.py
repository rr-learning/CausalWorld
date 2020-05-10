import pickle
import os
import json


class DataLoader:
    def __init__(self, episode_directory):
        if os.path.isdir(episode_directory):
            self.episode_directory = episode_directory
            info_path = os.path.join(self.episode_directory, "info.json")
            with open(info_path, "r") as json_file:
                info_dict = json.load(json_file)
                self.max_episode_index = info_dict["max_episode_index"]
                self.dumb_frequency = info_dict["dumb_frequency"]
        else:
            raise ValueError("data_path does not exist")

    def get_episodes(self, indices):
        episodes = []
        for index in indices:
            episodes.append(self.get_episode(index))

        return episodes

    def get_episode(self, index):
        if index > self.max_episode_index:
            raise Exception("Episode doesnt exist")
        infile_index_episode = index % self.dumb_frequency

        floor_index_episode = index - infile_index_episode
        ceil_index_episode = floor_index_episode + self.dumb_frequency - 1
        if ceil_index_episode > self.max_episode_index:
            ceil_index_episode = self.max_episode_index

        episode_file = "episode_{}_{}".format(floor_index_episode, ceil_index_episode)
        episodes_path = os.path.join(self.episode_directory, episode_file)
        if os.path.isfile(episodes_path):
            with open(episodes_path, "rb") as file:
                episodes = pickle.load(file)
                return episodes[infile_index_episode]
        else:
            raise Exception("Error: Log file with requested episode does not exist")
