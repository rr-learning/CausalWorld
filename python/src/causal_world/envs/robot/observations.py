import numpy as np
import math

from gym import spaces


class TriFingerObservations(object):

    def __init__(self,
                 observation_mode="structured",
                 normalize_observations=True,
                 observation_keys=None,
                 cameras=None,
                 camera_indicies=np.array([0, 1, 2])):
        """
        This class represents the observations limits of the robot and takes
        care of the normalization of the observation values.

        :param observation_mode:
        :param normalize_observations:
        :param observation_keys:
        :param cameras:
        :param camera_indicies:
        """

        num_fingers = 3
        self._normalized_observations = normalize_observations
        self._observation_mode = observation_mode
        self._camera_indicies = camera_indicies
        self._lower_bounds = dict()
        self._upper_bounds = dict()
        #TODO: add previous action space here!!!!

        self._lower_bounds["action_joint_positions"] = \
            [-1.57, -1.2, -3.0] * num_fingers
        self._upper_bounds["action_joint_positions"] = \
            [1.0, 1.57, 3.0] * num_fingers

        self._lower_bounds["end_effector_positions"] = \
            [-0.5, -0.5, 0.0] * num_fingers
        self._upper_bounds["end_effector_positions"] = \
            [0.5, 0.5, 0.5] * num_fingers

        self._lower_bounds["joint_torques"] = \
            [-0.36, -0.36, -0.36] * num_fingers
        self._upper_bounds["joint_torques"] = \
            [0.36, 0.36, 0.36] * num_fingers

        self._lower_bounds["joint_positions"] = \
            [-1.57, -1.2, -3.0] * num_fingers
        self._upper_bounds["joint_positions"] = \
            [1.0, 1.57, 3.0] * \
            num_fingers

        self._lower_bounds["joint_velocities"] = \
            [-50] * 3 * num_fingers
        self._upper_bounds["joint_velocities"] = \
            [50] * 3 * num_fingers

        num_of_cameras = self._camera_indicies.shape[0]
        self._lower_bounds["cameras"] = \
            np.zeros(shape=(num_of_cameras, 128, 128, 3), dtype=np.float64)
        self._upper_bounds["cameras"] = \
            np.full(shape=(num_of_cameras, 128, 128, 3), fill_value=255,
                    dtype=np.float64)

        self._observation_functions = dict()

        self._low_norm = -1
        self._high_norm = 1

        if observation_mode == "cameras":
            self._observations_keys = ["cameras"]
            self._low = np.zeros(shape=(num_of_cameras, 128, 128, 3),
                                 dtype=np.float64)
            self._high = np.full(shape=(num_of_cameras, 128, 128, 3),
                                 fill_value=255,
                                 dtype=np.float64)
            self._low_norm = 0
            self._high_norm = 1
            self._cameras = cameras
        elif observation_mode == "structured":
            if observation_keys is None:
                # Default structured observation space
                self._observations_keys = []
            elif all(key in self._lower_bounds.keys()
                     for key in observation_keys):
                self._observations_keys = observation_keys
            else:
                raise ValueError("One of the provided observation_"
                                 "keys is unknown")
            if self._observation_mode == "structured":
                self._observation_is_not_normalized = np.array([],
                                                               dtype=np.bool)
            self._low = np.array([])
            self._high = np.array([])
            self.set_observation_spaces()

    def get_observation_spaces(self):
        """

        :return:
        """
        if self._normalized_observations:
            observations_low_values = np.full(shape=self._low.shape,
                                              fill_value=self._low_norm,
                                              dtype=np.float64)
            observations_high_values = np.full(shape=self._low.shape,
                                               fill_value=self._high_norm,
                                               dtype=np.float64)
            if self._observation_mode == "structured":
                observations_low_values[self._observation_is_not_normalized] = \
                    self._low[self._observation_is_not_normalized]
                observations_high_values[self._observation_is_not_normalized] = \
                    self._high[self._observation_is_not_normalized]
            return spaces.Box(low=observations_low_values,
                              high=observations_high_values,
                              dtype=np.float64)
        else:
            if self._observation_mode == "structured":
                return spaces.Box(low=self._low,
                                  high=self._high,
                                  dtype=np.float64)
            else:
                return spaces.Box(low=self._low,
                                  high=self._high,
                                  dtype=np.uint8)

    def set_observation_spaces(self):
        """

        :return:
        """
        self._low = np.array([])
        self._high = np.array([])
        self._observation_is_not_normalized = np.array([], dtype=np.bool)
        if self._observation_mode == "cameras":
            self._low = np.array(self._lower_bounds['cameras'])
            self._high = np.array(self._lower_bounds['cameras'])
        else:
            for key in self._observations_keys:
                self._low = np.append(self._low,
                                      np.array(self._lower_bounds[key]))
                self._high = np.append(self._high,
                                       np.array(self._upper_bounds[key]))
                if np.array_equal(self._lower_bounds[key],
                                  self._upper_bounds[key]):
                    self._observation_is_not_normalized = \
                        np.append(self._observation_is_not_normalized,
                                  np.full(shape=
                                          np.array(
                                              self._upper_bounds[key]).shape,
                                          fill_value=True,
                                          dtype=np.bool))
                else:
                    self._observation_is_not_normalized = \
                        np.append(self._observation_is_not_normalized,
                                  np.full(shape=
                                          np.array(
                                              self._upper_bounds[key]).shape,
                                          fill_value=False,
                                          dtype=np.bool))
        return

    def is_normalized(self):
        """

        :return:
        """
        return self._normalized_observations

    def reset_observation_keys(self):
        """

        :return:
        """
        self._observations_keys = []
        self.set_observation_spaces()

    def normalize_observation(self, observation):
        """

        :param observation:

        :return:
        """
        return (self._high_norm - self._low_norm) * (observation - self._low) / \
               (self._high - self._low) \
               + self._low_norm

    def normalize_observation_for_key(self, observation, key):
        """

        :param observation:
        :param key:

        :return:
        """
        lower_key = np.array(self._lower_bounds[key])
        higher_key = np.array(self._upper_bounds[key])
        return (self._high_norm - self._low_norm) * (observation - lower_key) / \
               (higher_key - lower_key) + self._low_norm

    def denormalize_observation(self, observation):
        """

        :param observation:

        :return:
        """
        return self._low + (observation - self._low_norm) / \
               (self._high_norm - self._low_norm) * (self._high - self._low)

    def denormalize_observation_for_key(self, observation, key):
        """

        :param observation:
        :param key:

        :return:
        """
        lower_key = np.array(self._lower_bounds[key])
        higher_key = np.array(self._upper_bounds[key])
        return lower_key + (observation - self._low_norm) / \
               (self._high_norm - self._low_norm) * (higher_key - lower_key)

    def satisfy_constraints(self, observation):
        """

        :param observation:

        :return:
        """
        if self._normalized_observations:
            return (observation > self._low_norm).all() and \
                   (observation < self._high_norm).all()
        else:
            return (observation > self._low).all() and \
                   (observation < self._high).all()

    def clip_observation(self, observation):
        """

        :param observation:

        :return:
        """
        if self._normalized_observations:
            return np.clip(observation, self._low_norm, self._high_norm)
        else:
            return np.clip(observation, self._low, self._high)

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None, observation_fn=None):
        """

        :param observation_key:
        :param lower_bound:
        :param upper_bound:
        :param observation_fn:

        :return:
        """
        if observation_key not in self._lower_bounds.keys() and \
                (lower_bound is None or upper_bound is None):
            raise Exception("Observation key {} is not known please specify "
                            "the low and upper found".format(observation_key))
        if lower_bound is not None and upper_bound is not None:
            self._lower_bounds[observation_key] = lower_bound
            self._upper_bounds[observation_key] = upper_bound
        if observation_fn is not None:
            self._observation_functions[observation_key] = observation_fn
        self._observations_keys.append(observation_key)
        self.set_observation_spaces()

    def is_observation_key_known(self, observation_key):
        """

        :param observation_key:

        :return:
        """
        if observation_key not in self._lower_bounds.keys():
            return False
        else:
            return True

    def remove_observations(self, observations):
        """

        :param observations:

        :return:
        """
        for observation in observations:
            if observation not in self._observations_keys:
                raise Exception(
                    "Observation key {} is not known".format(observation))
            self._observations_keys.remove(observation)
        self.set_observation_spaces()

    #since task might need access to state elements that are not in the observation leys
    def get_current_observations(self, robot_state, helper_keys):
        """

        :param robot_state:
        :param helper_keys:

        :return:
        """
        observations_dict = dict()
        for observation in self._observations_keys:
            if observation == "joint_positions":
                observations_dict["joint_positions"] = \
                    robot_state['positions']
            elif observation == "joint_torques":
                observations_dict["joint_torques"] = \
                    robot_state['torques']
            elif observation == "joint_velocities":
                observations_dict["joint_velocities"] = \
                    robot_state['velocities']
            elif observation == "end_effector_positions":
                observations_dict["end_effector_positions"] = \
                    robot_state['end_effector_positions']
            elif observation == "cameras":
                camera_obs = np.stack(
                    (self._cameras[0].get_image(), self._cameras[1].get_image(),
                     self._cameras[2].get_image()),
                    axis=0)
                observations_dict["cameras"] = camera_obs
            elif observation in self._observation_functions:
                observations_dict[observation] = \
                    self._observation_functions[observation](robot_state)
            #else its in the spaces but will be handled by the task itself,
            # the task also needs to check for normalization somehow
        #add the helper observations as well
        for observation in helper_keys:
            if observation == "joint_positions":
                observations_dict["joint_positions"] = \
                    robot_state['positions']
            elif observation == "joint_torques":
                observations_dict["joint_torques"] = \
                    robot_state['torques']
            elif observation == "joint_velocities":
                observations_dict["joint_velocities"] = \
                    robot_state['velocities']
            elif observation == "end_effector_positions":
                observations_dict["end_effector_positions"] = \
                    robot_state['end_effector_positions']
            elif observation == "cameras":
                images = []
                for i in self._camera_indicies:
                    images.append(self._cameras[i].get_image())
                camera_obs = np.stack(images, axis=0)
                observations_dict["cameras"] = camera_obs
            elif observation in self._observation_functions:
                observations_dict[observation] = \
                    self._observation_functions[observation](robot_state)
            else:
                raise Exception("The robot doesn't know about observation "
                                "key {}".format(observation))
        #now normalize everything here
        if self._normalized_observations:
            for key in observations_dict.keys():
                observations_dict[key] = \
                    self.normalize_observation_for_key(observations_dict[key],
                                                       key)

        return observations_dict

    def get_current_camera_observations(self):
        """

        :return:
        """
        images = []
        for i in self._camera_indicies:
            images.append(self._cameras[i].get_image())
        camera_obs = np.stack(images, axis=0)
        if self._normalized_observations:
            camera_obs = self.normalize_observation_for_key(
                camera_obs, "cameras")
        return camera_obs
