import numpy as np
import math

import gym
from gym import spaces


class SingleCameraObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(540, 720, 3),
                                                dtype=np.uint8)

    def observation(self, observation):
        return self.env.latest_observation.camera_300


class AllCameraObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(3, 540, 720, 3),
                                                dtype=np.uint8)

    def observation(self, observation):
        latest_camera_observations = [self.env.latest_observation.camera_60,
                                      self.env.latest_observation.camera_180,
                                      self.env.latest_observation.camera_300]
        return np.stack(latest_camera_observations, axis=0)


class TriFingerStructuredObservationSpaces:
    """
    Sets up the observation and action spaces for a finger env depending
    on the observations used.

    Args:
        observations_keys (list of strings): The keys corresponding to the
            observations used in the env
        observations_sizes (list of ints): The sizes of each of the keys
            in observations_keys in the same order.
    """

    def __init__(
        self,
        observations_keys,
        observations_sizes,
    ):

        self.num_fingers = 3

        self.lower_bounds = {}
        self.upper_bounds = {}

        self.observations_keys = observations_keys
        self.observations_sizes = observations_sizes
        self.key_to_index = {}

        assert len(self.observations_keys) == len(self.observations_sizes), (
            "Specify the size for each expected observation key."
            "And this is not being checked, but the sizes must be"
            "in the same order as the keys."
        )
        if len(self.observations_keys) == 0:
            self.observations_keys = [
                "joint_positions",
                "joint_velocities",
                "action_joint_positions",
                "end_effector_positions"
            ]
            self.observations_sizes = [
                3 * self.num_fingers,
                3 * self.num_fingers,
                3 * self.num_fingers,
                3 * self.num_fingers
            ]

        slice_start = 0
        for i in range(len(self.observations_keys)):
            self.key_to_index[self.observations_keys[i]] = slice(
                slice_start, slice_start + self.observations_sizes[i]
            )
            slice_start += self.observations_sizes[i]

        self.lower_bounds["action_joint_positions"] = np.array(
                [-math.radians(70), -math.radians(70), -math.radians(160)]
                * self.num_fingers
            )
        self.upper_bounds["action_joint_positions"] = np.array(
                [math.radians(70), 0, math.radians(-2)] * self.num_fingers
            )

        self.lower_bounds["end_effector_positions"] = [
            -0.5,
            -0.5,
            0.0,
        ] * self.num_fingers
        self.upper_bounds["end_effector_positions"] = [
            0.5,
            0.5,
            0.5,
        ] * self.num_fingers

        self.lower_bounds["joint_positions"] = [
            -math.radians(90),
            -math.radians(90),
            -math.radians(172),
        ] * self.num_fingers
        self.upper_bounds["joint_positions"] = [
            math.radians(90),
            math.radians(100),
            math.radians(-2),
        ] * self.num_fingers

        self.lower_bounds["joint_velocities"] = [-20] * 3 * self.num_fingers
        self.upper_bounds["joint_velocities"] = [20] * 3 * self.num_fingers

    def get_unscaled_observation_space(self):
        """
        Returns the unscaled observation space corresponding
        to the observation bounds
        """
        observation_lower_bounds = [
            value
            for key in self.observations_keys
            for value in self.lower_bounds[key]
        ]
        observation_higher_bounds = [
            value
            for key in self.observations_keys
            for value in self.upper_bounds[key]
        ]
        return spaces.Box(
            low=np.array(observation_lower_bounds),
            high=np.array(observation_higher_bounds),
            dtype=np.float64
        )

    def get_scaled_observation_space(self):
        """
        Returns an observation space with the same size as the unscaled
        but bounded by -1s and 1s.
        """
        unscaled_observation_space = self.get_unscaled_observation_space()
        return spaces.Box(
            low=-np.ones_like(unscaled_observation_space.low),
            high=np.ones_like(unscaled_observation_space.high),
            dtype=np.float64
        )

    def add_scene_object(self, so_observation_keys, so_observation_sizes):
        assert len(so_observation_keys) == len(so_observation_sizes)
        # self.observations_keys.extend(so_observation_keys)
        # self.observations_sizes.extend(so_observation_sizes)

        # Bounds of scene object position
        self.lower_bounds[so_observation_keys[0]] = [-0.5] * 3
        self.upper_bounds[so_observation_keys[0]] = [0.5] * 3

        # Bounds of scene object orientation
        # TODO: Check if these bounds make sense for orientation quaternions
        self.lower_bounds[so_observation_keys[1]] = [-1.0] * 4
        self.upper_bounds[so_observation_keys[1]] = [1.0] * 4

        # Bounds of scene object id, for now this is stored as a continuous variable
        self.lower_bounds[so_observation_keys[2]] = [0.0]
        self.upper_bounds[so_observation_keys[2]] = [100.0]

        # Update key_to_index dict
        slice_start = 0
        for i in range(len(self.observations_keys)):
            self.key_to_index[self.observations_keys[i]] = slice(
                slice_start, slice_start + self.observations_sizes[i]
            )
            slice_start += self.observations_sizes[i]