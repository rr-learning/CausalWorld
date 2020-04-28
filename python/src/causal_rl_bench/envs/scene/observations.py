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

        self.rigid_objects = rigid_objects
        self.visual_objects = visual_objects

        self.observations_keys = []
        self.low = np.array([])
        self.high = np.array([])
        self.visual_objects_state = dict()
        self.set_observation_spaces()
        self.initialize_observations()
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

    def set_observation_spaces(self):
        self.low = np.array([])
        self.high = np.array([])
        for key in self.observations_keys:
            self.low = np.append(self.low, np.array(self.lower_bounds[key]))
            self.high = np.append(self.high, np.array(self.upper_bounds[key]))

    def is_normalized(self):
        return self.normalized_observations

    def normalize_observation(self, observation):
        return 2.0 * (observation - self.low) / (self.high - self.low) - 1.0

    def denormalize_observation(self, observation):
        return self.low + (observation + 1.0) / 2.0 * (self.high - self.low)

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

    def get_current_observations(self):
        # TODO: scale if normalized
        observations_dict = dict()
        for rigid_object in self.rigid_objects:
            observations_dict.update(rigid_object.get_state())
        observations_dict.update(self.visual_objects_state)
        return observations_dict

    def remove_observations(self, observations):
        for observation in observations:
            if observation not in self.observations_keys:
                raise Exception(
                    "Observation key {} is not known".format(observation))
            del self.lower_bounds[observation]
            del self.upper_bounds[observation]
            self.observations_keys.remove(observation)
        self.set_observation_spaces()



