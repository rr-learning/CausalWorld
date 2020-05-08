import numpy as np
from gym import spaces


class StageObservations(object):
    def __init__(self, rigid_objects, visual_objects,
                 observation_mode="structured",
                 normalize_observations=True):
        self.normalized_observations = normalize_observations
        self.observation_mode = observation_mode

        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds["goal_image"] = \
            np.zeros(shape=(3, 540, 720, 3), dtype=np.float64)
        self.upper_bounds["goal_image"] = \
            np.full(shape=(3, 540, 720, 3), fill_value=255,
                    dtype=np.float64)

        self.rigid_objects = rigid_objects
        self.visual_objects = visual_objects

        self.observations_keys = []
        self.low = np.array([])
        self.high = np.array([])
        self.visual_objects_state = dict()
        self.initialize_observations()
        self.set_observation_spaces()
        return

    def get_observation_spaces(self):
        if self.normalized_observations:
            return spaces.Box(low=-np.ones(shape=self.low.shape),
                              high=np.ones(shape=self.high.shape),
                              dtype=np.float64)
        else:
            if self.observation_mode == "structured":
                return spaces.Box(low=self.low,
                                  high=self.high,
                                  dtype=np.float64)
            else:
                return spaces.Box(low=self.low,
                                  high=self.high,
                                  dtype=np.uint8)

    def initialize_observations(self):
        for rigid_object in self.rigid_objects:
            state_keys = rigid_object.get_state().keys()
            for state_key in state_keys:
                self.lower_bounds[state_key] = \
                    rigid_object.lower_bounds[state_key]
                self.upper_bounds[state_key] = \
                    rigid_object.upper_bounds[state_key]
                self.observations_keys.append(state_key)
        for visual_object in self.visual_objects:
            self.visual_objects_state.update(visual_object.get_state())
            state_keys = visual_object.get_state().keys()
            for state_key in state_keys:
                self.lower_bounds[state_key] = \
                    visual_object.lower_bounds[state_key]
                self.upper_bounds[state_key] = \
                    visual_object.upper_bounds[state_key]
                self.observations_keys.append(state_key)
        return

    def reset_observation_keys(self):
        self.observations_keys = []
        self.set_observation_spaces()

    def is_observation_key_known(self, observation_key):
        if observation_key not in self.lower_bounds.keys():
            return False
        else:
            return True

    def set_observation_spaces(self):
        self.low = np.array([])
        self.high = np.array([])
        if "goal_image" in self.observations_keys:
            self.low = self.lower_bounds["goal_image"]
            self.high = self.upper_bounds["goal_image"]
        else:
            for key in self.observations_keys:
                self.low = np.append(self.low,
                                     np.array(self.lower_bounds[key]))
                self.high = np.append(self.high,
                                      np.array(self.upper_bounds[key]))

    def is_normalized(self):
        return self.normalized_observations

    def normalize_observation(self, observation):
        return 2.0 * (observation - self.low) / (self.high - self.low) - 1.0

    def denormalize_observation(self, observation):
        return self.low + (observation + 1.0) / 2.0 * (self.high - self.low)

    def normalize_observation_for_key(self, observation, key):
        lower_key = np.array(self.lower_bounds[key])
        higher_key = np.array(self.upper_bounds[key])
        return 2.0 * (observation - lower_key) / (higher_key - lower_key) - 1.0

    def denormalize_observation_for_key(self, observation, key):
        lower_key = np.array(self.lower_bounds[key])
        higher_key = np.array(self.upper_bounds[key])
        return lower_key + (observation + 1.0) / 2.0 * (higher_key - lower_key)

    def satisfy_constraints(self, observation):
        if self.normalized_observations:
            return (observation > -1.).all() and (observation < 1.).all()
        else:
            return (observation > self.low).all() and \
                   (observation < self.high).all()

    def clip_observation(self, observation):
        if self.normalized_observations:
            return np.clip(observation, -1.0, 1.0)
        else:
            return np.clip(observation, self.low, self.high)

    def get_current_observations(self, helper_keys):
        observations_dict = dict()
        for rigid_object in self.rigid_objects:
            observations_dict.update(rigid_object.get_state())
        observations_dict.update(self.visual_objects_state)
        observation_dict_keys = list(observations_dict.keys())
        for observation in observation_dict_keys:
            if (observation not in self.observations_keys) and (observation not in helper_keys):
                del observations_dict[observation]
        # now normalize everything here
        if self.normalized_observations:
            for key in observations_dict.keys():
                observations_dict[key] = self.normalize_observation_for_key(observations_dict[key], key)
        return observations_dict

    def remove_observations(self, observations):
        for observation in observations:
            if observation not in self.observations_keys:
                raise Exception(
                    "Observation key {} is not known".format(observation))
            self.observations_keys.remove(observation)
        self.set_observation_spaces()

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None):
        if observation_key not in self.lower_bounds.keys() and \
                (lower_bound is None or upper_bound is None):
            raise Exception("Observation key {} is not known please specify "
                            "the low and upper found".format(observation_key))
        if lower_bound is not None and upper_bound is not None:
            self.lower_bounds[observation_key] = lower_bound
            self.upper_bounds[observation_key] = upper_bound
        self.observations_keys.append(observation_key)
        self.set_observation_spaces()

    def get_current_goal_image(self, goal_instance_state):
        camera_obs = np.stack((goal_instance_state.camera_60,
                               goal_instance_state.camera_180,
                               goal_instance_state.camera_300), axis=0)
        if self.normalized_observations:
            camera_obs = self.normalize_observation_for_key(camera_obs,
                                                            "goal_image")
        return camera_obs


