from causal_world.loggers.episode import Episode

import json
import pickle
import os


class DataRecorder:
    def __init__(self, output_directory=None, rec_dumb_frequency=100):
        """
        This class logs the full histories of a world across multiple episodes

        :param output_directory:
        :param rec_dumb_frequency:
        """
        self.rec_dumb_frequency = rec_dumb_frequency
        # if output_directory is None:
        #     self.path = os.path.join("output", "logs")
        #     if not os.path.isdir(self.path):
        #         os.makedirs(self.path)
        # else:
        if output_directory is not None:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
        self.path = output_directory
        self.episodes = []
        self.last_episode_number_dumbed = len(self.episodes) - 1
        self._curr = None

    def new_episode(self,
                    initial_full_state,
                    task_name,
                    task_params=None,
                    world_params=None):
        """

        :param initial_full_state:
        :param task_name:
        :param task_params:
        :param world_params:
        :return:
        """
        if self._curr:
            self.episodes.append(self._curr)
        self._curr = Episode(task_name,
                             initial_full_state,
                             task_params=task_params,
                             world_params=world_params)
        if self.path is not None and \
                len(self.episodes) % self.rec_dumb_frequency == 0 and len(self.episodes) != 0:
            self.save()

    def append(self, robot_action, observation, reward, info, done, timestamp):
        """

        :param robot_action:
        :param observation:
        :param reward:
        :param info:
        :param done:
        :param timestamp:

        :return:
        """
        self._curr.append(robot_action, observation, reward, info, done,
                          timestamp)

    def save(self):
        """

        :return:
        """
        if self.path is None:
            return
        if len(self._curr.observations):
            self.episodes.append(self._curr)
        new_episode_number_dumbed = self.last_episode_number_dumbed + len(
            self.episodes)
        file_path = os.path.join(
            self.path,
            "episode_{}_{}".format(self.last_episode_number_dumbed + 1,
                                   new_episode_number_dumbed))
        with open(file_path, "wb") as file_handle:
            pickle.dump(self.episodes, file_handle)
            self.last_episode_number_dumbed = new_episode_number_dumbed
            self.episodes = []

        info_path = os.path.join(self.path, "info.json")
        with open(info_path, "w") as json_file:
            info_dict = {
                "dumb_frequency": self.rec_dumb_frequency,
                "max_episode_index": new_episode_number_dumbed
            }
            json.dump(info_dict, json_file)

    def get_number_of_logged_episodes(self):
        """

        :return:
        """
        return self.last_episode_number_dumbed + len(self.episodes) + 1

    def get_current_episode(self):
        """

        :return:
        """
        return self._curr

    def clear_recorder(self):
        """

        :return:
        """
        self.episodes = []
