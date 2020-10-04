from causal_world.loggers.episode import Episode

import json
import pickle
import os


class DataRecorder:
    def __init__(self, output_directory=None, rec_dumb_frequency=100):
        """
        This class logs the full histories of a world across multiple episodes

        :param output_directory: (str) specifies the output directory to save
                                       the episodes in.
        :param rec_dumb_frequency: (int) specifies the peridicity of saving
                                         the episodes.
        """
        self.rec_dumb_frequency = rec_dumb_frequency
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

        :param initial_full_state: (dict) dict specifying the full state
                                          variables of the environment.
        :param task_name: (str) task generator name.
        :param task_params: (dict) task generator parameters.
        :param world_params: (dict) causal world parameters.
        :return:
        """
        if self._curr:
            self.episodes.append(self._curr)
        self._curr = Episode(task_name,
                             initial_full_state,
                             task_params=task_params,
                             world_params=world_params)
        if self.path is not None and \
                len(self.episodes) % self.rec_dumb_frequency == 0 and \
                len(self.episodes) != 0:
            self.save()
        return

    def append(self, robot_action, observation, reward, info, done, timestamp):
        """

        :param robot_action: (nd.array) action passed to step function.
        :param observation: (nd.array) observations returned after stepping
                                       through the environment.
        :param reward: (float) reward received from the environment.
        :param info: (dict) dictionary specifying all the extra information
                            after stepping through the environment.
        :param done: (bool) true if the environment returns done.
        :param timestamp: (float) time stamp with respect to the beginning of
                                  the episode.

        :return:
        """
        self._curr.append(robot_action, observation, reward, info, done,
                          timestamp)
        return

    def save(self):
        """
        dumps the current episodes.

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

        :return: (int) number of logged episodes.
        """
        return self.last_episode_number_dumbed + len(self.episodes) + 1

    def get_current_episode(self):
        """

        :return: (causal_world.loggers.Episode) current episode saved.
        """
        return self._curr

    def clear_recorder(self):
        """
        Clears the data recorder.

        :return:
        """
        self.episodes = []
