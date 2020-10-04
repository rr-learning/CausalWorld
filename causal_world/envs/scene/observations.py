import numpy as np
from gym import spaces


class StageObservations(object):

    def __init__(self,
                 rigid_objects,
                 visual_objects,
                 observation_mode="structured",
                 normalize_observations=True,
                 cameras=None,
                 camera_indicies=np.array([0, 1, 2])):
        """

        :param rigid_objects: (dict) dict of rigid objects in the arena.
        :param visual_objects: (dict) dict of visual objects in the arena.
        :param observation_mode: (str) specifies the observation mode
                                       if structured or cameras.
        :param normalize_observations: (bool) specifies if the observations are
                                              normalized or not.
        :param cameras: (list) a list of cameras mounted on the stage.
        :param camera_indicies: (list) specifies the indicies of the cameras
                                       to be specified.
        """
        self._normalized_observations = normalize_observations
        self._observation_mode = observation_mode
        self._camera_indicies = camera_indicies
        self._low_norm = -1
        self._high_norm = 1
        if observation_mode == "pixel":
            self._low_norm = 0
            self._high_norm = 1

        self._lower_bounds = dict()
        self._upper_bounds = dict()
        num_of_cameras = self._camera_indicies.shape[0]
        self._lower_bounds["goal_image"] = \
            np.zeros(shape=(num_of_cameras, 128, 128, 3), dtype=np.float64)
        self._upper_bounds["goal_image"] = \
            np.full(shape=(num_of_cameras, 128, 128, 3), fill_value=255,
                    dtype=np.float64)
        self._goal_cameras = cameras

        self._rigid_objects = rigid_objects
        self._visual_objects = visual_objects

        self._observations_keys = []
        self._observation_is_not_normalized = np.array([], dtype=np.bool)
        self._low = np.array([])
        self._high = np.array([])
        self.initialize_observations()
        self.set_observation_spaces()
        return

    def get_observation_spaces(self):
        """

        :return: (gym.Spaces) observation space as a gym box space.
        """
        if self._normalized_observations:
            observations_low_values = np.full(shape=self._low.shape,
                                              fill_value=self._low_norm,
                                              dtype=np.float64)
            observations_high_values = np.full(shape=self._low.shape,
                                               fill_value=self._high_norm,
                                               dtype=np.float64)
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

    def initialize_observations(self):
        """
        Creates the upper bound and lower bound of the observation space.

        :return:
        """
        for rigid_object in self._rigid_objects.values():
            state_keys = rigid_object.get_state().keys()
            object_lower_bounds, object_upper_bounds = \
                rigid_object.get_bounds()
            for state_key in state_keys:

                self._lower_bounds[rigid_object.get_name() + '_' + state_key] = \
                    object_lower_bounds[rigid_object.get_name() + '_' +
                                        state_key]
                self._upper_bounds[rigid_object.get_name() + '_' + state_key] = \
                    object_upper_bounds[rigid_object.get_name() + '_' +
                                        state_key]
        for visual_object in self._visual_objects.values():
            state_keys = visual_object.get_state().keys()
            object_lower_bounds, object_upper_bounds = \
                visual_object.get_bounds()
            for state_key in state_keys:
                self._lower_bounds[visual_object.get_name() + '_' + state_key] = \
                    object_lower_bounds[visual_object.get_name() + '_' +
                                               state_key]
                self._upper_bounds[visual_object.get_name() + '_' + state_key] = \
                    object_upper_bounds[visual_object.get_name() + '_' +
                                               state_key]
        return

    def reset_observation_keys(self):
        """
        Resets the observation keys and the lower bound as well as upper bound.

        :return:
        """
        self._observations_keys = []
        self._low = np.array([])
        self._high = np.array([])

    def _is_observation_key_known(self, observation_key):
        """

        :param observation_key: (str) specifies the observation key to query.

        :return: (bool) returns true if the observation key is
                        known to the space.
        """
        if observation_key not in self._lower_bounds.keys():
            return False
        else:
            return True

    def set_observation_spaces(self):
        """
        sets the lower and upper bound on the observation space.

        :return:
        """
        self._low = np.array([])
        self._high = np.array([])
        self._observation_is_not_normalized = np.array([], dtype=np.bool)
        if "goal_image" in self._observations_keys:
            self._low = self._lower_bounds["goal_image"]
            self._high = self._upper_bounds["goal_image"]
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

        :return: (bool) returns true if the observations are normalized or not.
        """
        return self._normalized_observations

    def normalize_observation(self, observation):
        """

        :param observation: (nd.array) represents the observation to
                                       be normalized.

        :return: (nd.array) normalized observation.
        """
        return (self._high_norm - self._low_norm) * \
               (observation - self._low) / (self._high - self._low) \
               + self._low_norm

    def denormalize_observation(self, observation):
        """

        :param observation: (nd.array) represents the observation to
                                       be denormalized.

        :return: (nd.array) denormalized observation.
        """
        return self._low + (observation - self._low_norm) / (self._high_norm -
                                                             self._low_norm) * \
               (self._high - self._low)

    def normalize_observation_for_key(self, observation, key):
        """

        :param observation: (nd.array) observation vector to normalize.
        :param key: (str) key corresponding to the observation vector.

        :return: (nd.array) normalized observation vector.
        """
        lower_key = np.array(self._lower_bounds[key])
        higher_key = np.array(self._upper_bounds[key])
        if np.array(lower_key == higher_key).all():
            return observation
        return (self._high_norm - self._low_norm) * (
            observation - lower_key) / (higher_key - lower_key) + self._low_norm

    def denormalize_observation_for_key(self, observation, key):
        """
        :param observation: (nd.array) observation vector to denormalize.
        :param key: (str) key corresponding to the observation vector.

        :return: (nd.array) denormalized observation vector.
        """
        lower_key = np.array(self._lower_bounds[key])
        higher_key = np.array(self._upper_bounds[key])
        if np.array(lower_key == higher_key).all():
            return observation
        return lower_key + (observation - self._low_norm) / (
            self._high_norm - self._low_norm) * (higher_key - lower_key)

    def satisfy_constraints(self, observation):
        """

        :param observation: (nd.array) observation vector to check if it
                                       satisfies the constraints.

        :return: (bool) returns true if the constraints are satisified, false
                        otherwise.
        """
        if self._normalized_observations:
            return (observation > self._low_norm).all() and (
                observation < self._high_norm).all()
        else:
            return (observation > self._low).all() and \
                   (observation < self._high).all()

    def clip_observation(self, observation):
        """

        :param observation: (nd.array) observation vector to clip.

        :return: (nd.array) clipped observation vector to satisfy the limits.
        """
        if self._normalized_observations:
            return np.clip(observation, self._low_norm, self._high_norm)
        else:
            return np.clip(observation, self._low, self._high)

    def get_current_observations(self, helper_keys):
        """
        :param helper_keys: (list) observation keys that are needed but not in
                                   the observation space for further calculation
                                   of custom observations or reward function
                                   calculation.

        :return: (dict) returns a dict for all the observation keys and helper
                        keys as well to be processed accordingly. Also
                        normalization takes effect here if needed.
        """
        observations_dict = dict()
        for rigid_object in self._rigid_objects.values():
            observations_dict.update({
                rigid_object.get_name() + '_' + k: v
                for k, v in rigid_object.get_state().items()
            })
        for visual_object in self._visual_objects.values():
            observations_dict.update({
                visual_object.get_name() + '_' + k: v
                for k, v in visual_object.get_state().items()
            })
        observation_dict_keys = list(observations_dict.keys())
        for observation in observation_dict_keys:
            if (observation not in self._observations_keys) and \
                    (observation not in helper_keys):
                del observations_dict[observation]
        # now normalize everything here
        if self._normalized_observations:
            for key in observations_dict.keys():
                observations_dict[key] = \
                    self.normalize_observation_for_key(
                        observations_dict[key], key)
        return observations_dict

    def remove_observations(self, observations):
        """

        :param observations: (list) list of observation keys to remove from
                                    the observation space.

        :return: None
        """
        for observation in observations:
            if observation not in self._observations_keys:
                raise Exception(
                    "Observation key {} is not known".format(observation))
            self._observations_keys.remove(observation)
        self.set_observation_spaces()

    def add_observation(self,
                        observation_key,
                        lower_bound=None,
                        upper_bound=None):
        """

        :param observation_key: (str) observation key to be added.
        :param lower_bound: (nd.array) lower bound corresponding to the
                                       observation key if not known.
        :param upper_bound: (nd.array) upper bound corresponding to the
                                       observation key if not known.

        :return: None
        """
        if observation_key not in self._lower_bounds.keys() and \
                (lower_bound is None or upper_bound is None):
            raise Exception("Observation key {} is not known please specify "
                            "the low and upper found".format(observation_key))
        if lower_bound is not None and upper_bound is not None:
            self._lower_bounds[observation_key] = lower_bound
            self._upper_bounds[observation_key] = upper_bound
        self._observations_keys.append(observation_key)
        self.set_observation_spaces()

    def get_current_goal_image(self):
        """

        :return: (nd.array) returns observations from the cameras in the goal
                            pybullet instance if in "pixel" mode,
                            normalization takes place here.
        """
        images = []
        for i in self._camera_indicies:
            images.append(self._goal_cameras[i].get_image())
        camera_obs = np.stack(images, axis=0)
        if self._normalized_observations:
            camera_obs = self.normalize_observation_for_key(
                camera_obs, "goal_image")
        return camera_obs
