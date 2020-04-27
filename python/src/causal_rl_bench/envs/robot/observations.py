import numpy as np
import math

from gym import spaces


class TriFingerObservations(object):
    def __init__(self, observation_mode="structured", normalize_observations=True,
                 observation_keys=None):

        self.num_fingers = 3
        self.normalized_observations = normalize_observations
        self.observation_mode = observation_mode

        self.lower_bounds = {}
        self.upper_bounds = {}

        self.lower_bounds["action_joint_positions"] = [-math.radians(70),
                                                       -math.radians(70),
                                                       -math.radians(160)] * self.num_fingers
        self.upper_bounds["action_joint_positions"] = [math.radians(70),
                                                       math.radians(0),
                                                       math.radians(-2)] * self.num_fingers

        self.lower_bounds["end_effector_positions"] = [-0.5, -0.5, 0.0] * self.num_fingers
        self.upper_bounds["end_effector_positions"] = [0.5, 0.5, 0.5] * self.num_fingers

        self.lower_bounds["joint_torques"] = [-0.36, -0.36, -0.36] * self.num_fingers
        self.upper_bounds["joint_torques"] = [0.36, 0.36, 0.36] * self.num_fingers

        self.lower_bounds["joint_positions"] = [-math.radians(90),
                                                -math.radians(90),
                                                -math.radians(172)] * self.num_fingers
        self.upper_bounds["joint_positions"] = [math.radians(90),
                                                math.radians(100),
                                                math.radians(-2)] * self.num_fingers

        self.lower_bounds["joint_velocities"] = [-20] * 3 * self.num_fingers
        self.upper_bounds["joint_velocities"] = [20] * 3 * self.num_fingers

        if observation_mode == "cameras":
            self.low = np.zeros(shape=(3, 540, 720, 3), dtype=np.uint8)
            self.high = np.full(shape=(3, 540, 720, 3), fill_value=255, dtype=np.uint8)
        elif observation_mode == "structured":
            if observation_keys is None:
                # Default structured observation space
                self.observations_keys = ["joint_positions",
                                          "joint_velocities",
                                          "joint_torques"]
            elif all(key in observation_keys for key in self.lower_bounds.keys()):
                self.observation_keys = observation_keys
            else:
                raise ValueError("One of the provided observation_keys is unknown")

            self.low = np.array([])
            self.high = np.array([])
            for key in self.observations_keys:
                self.low = np.append(self.low, np.array(self.lower_bounds[key]))
                self.high = np.append(self.high, np.array(self.upper_bounds[key]))

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
            return (observation > self.low).all() and (observation < self.high).all()

    def clip_observation(self, observation):
        if self.normalized_observations:
            return np.clip(observation, -1.0, 1.0)
        else:
            return np.clip(observation, self.low, self.high)



