import numpy as np
import math

from gym import spaces


class TriFingerObservations(object):
    def __init__(self, observation_mode="structured",
                 normalize_observations=True,
                 observation_keys=None):

        num_fingers = 3
        self.normalized_observations = normalize_observations
        self.observation_mode = observation_mode

        self.lower_bounds = dict()
        self.upper_bounds = dict()
        #TODO: add previous action space here!!!!
        self.lower_bounds["action_joint_positions"] = \
            [-math.radians(70), -math.radians(70),
             -math.radians(160)] * num_fingers
        self.upper_bounds["action_joint_positions"] = \
            [math.radians(70), 0,
             math.radians(-2)] * num_fingers

        self.lower_bounds["end_effector_positions"] = \
            [-0.5, -0.5, 0.0] * num_fingers
        self.upper_bounds["end_effector_positions"] = \
            [0.5, 0.5, 0.5] * num_fingers

        self.lower_bounds["joint_torques"] = \
            [-0.36, -0.36, -0.36] * num_fingers
        self.upper_bounds["joint_torques"] = \
            [0.36, 0.36, 0.36] * num_fingers

        self.lower_bounds["joint_positions"] = \
            [-math.radians(70), -math.radians(70),
             -math.radians(160)] * num_fingers
        self.upper_bounds["joint_positions"] = \
            [math.radians(70),
             0, math.radians(-2)] * \
            num_fingers

        self.lower_bounds["joint_velocities"] = \
            [-20] * 3 * num_fingers
        self.upper_bounds["joint_velocities"] = \
            [20] * 3 * num_fingers

        self.lower_bounds["cameras"] = \
            np.zeros(shape=(3, 54, 72, 3), dtype=np.float64)
        self.upper_bounds["cameras"] = \
            np.full(shape=(3, 54, 72, 3), fill_value=255,
                    dtype=np.float64)

        self.observation_functions = dict()

        self.low_norm = -1
        self.high_norm = 1

        if observation_mode == "cameras":
            self.observations_keys = ["cameras"]
            self.low = np.zeros(shape=(3, 54, 72, 3), dtype=np.float64)
            self.high = np.full(shape=(3, 54, 72, 3), fill_value=255,
                                dtype=np.float64)
            self.low_norm = 0
            self.high_norm = 1
        elif observation_mode == "structured":
            if observation_keys is None:
                # Default structured observation space
                self.observations_keys = []
            elif all(key in self.lower_bounds.keys()
                     for key in observation_keys):
                self.observations_keys = observation_keys
            else:
                raise ValueError("One of the provided observation_"
                                 "keys is unknown")

            self.low = np.array([])
            self.high = np.array([])
            self.set_observation_spaces()

    def get_observation_spaces(self):
        if self.normalized_observations:
            return spaces.Box(low=np.full(shape=self.low.shape,
                                          fill_value=self.low_norm,
                                          dtype=np.float64),
                              high=np.full(shape=self.low.shape,
                                           fill_value=self.high_norm,
                                           dtype=np.float64),
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

    def set_observation_spaces(self):
        self.low = np.array([])
        self.high = np.array([])
        for key in self.observations_keys:
            self.low = np.append(self.low, np.array(self.lower_bounds[key]))
            self.high = np.append(self.high, np.array(self.upper_bounds[key]))

    def is_normalized(self):
        return self.normalized_observations

    def reset_observation_keys(self):
        self.observations_keys = []
        self.set_observation_spaces()

    def normalize_observation(self, observation):
        return (self.high_norm - self.low_norm) * (observation - self.low) / \
               (self.high - self.low) \
               + self.low_norm

    def normalize_observation_for_key(self, observation, key):
        lower_key = np.array(self.lower_bounds[key])
        higher_key = np.array(self.upper_bounds[key])
        return (self.high_norm - self.low_norm) * (observation - lower_key) / \
               (higher_key - lower_key) + self.low_norm

    def denormalize_observation(self, observation):
        return self.low + (observation - self.low_norm) / \
               (self.high_norm - self.low_norm) * (self.high - self.low)

    def denormalize_observation_for_key(self, observation, key):
        lower_key = np.array(self.lower_bounds[key])
        higher_key = np.array(self.upper_bounds[key])
        return lower_key + (observation - self.low_norm) / \
               (self.high_norm - self.low_norm) * (higher_key - lower_key)

    def satisfy_constraints(self, observation):
        if self.normalized_observations:
            return (observation > self.low_norm).all() and \
                   (observation < self.high_norm).all()
        else:
            return (observation > self.low).all() and \
                   (observation < self.high).all()

    def clip_observation(self, observation):
        if self.normalized_observations:
            return np.clip(observation, self.low_norm, self.high_norm)
        else:
            return np.clip(observation, self.low, self.high)

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None, observation_fn=None):
        if observation_key not in self.lower_bounds.keys() and \
                (lower_bound is None or upper_bound is None):
            raise Exception("Observation key {} is not known please specify "
                            "the low and upper found".format(observation_key))
        if lower_bound is not None and upper_bound is not None:
            self.lower_bounds[observation_key] = lower_bound
            self.upper_bounds[observation_key] = upper_bound
        if observation_fn is not None:
            self.observation_functions[observation_key] = observation_fn
        self.observations_keys.append(observation_key)
        self.set_observation_spaces()

    def is_observation_key_known(self, observation_key):
        if observation_key not in self.lower_bounds.keys():
            return False
        else:
            return True

    def remove_observations(self, observations):
        for observation in observations:
            if observation not in self.observations_keys:
                raise Exception(
                    "Observation key {} is not known".format(observation))
            self.observations_keys.remove(observation)
        self.set_observation_spaces()

    #since task might need access to state elements that are not in the observation leys
    def get_current_observations(self, robot_state, helper_keys):
        observations_dict = dict()
        for observation in self.observations_keys:
            if observation == "joint_positions":
                observations_dict["joint_positions"] = \
                    robot_state['positions']
            elif observation == "joint_torques":
                observations_dict["joint_torques"] = \
                    robot_state['torques']
            elif observation == "joint_velocities":
                observations_dict["joint_velocities"] = \
                    robot_state['velocities']
            elif observation == "cameras":
                camera_obs = np.stack((robot_state.camera_60,
                                      robot_state.camera_180,
                                      robot_state.camera_300), axis=0)
                observations_dict["cameras"] = camera_obs
            elif observation in self.observation_functions:
                observations_dict[observation] = \
                    self.observation_functions[observation](robot_state)
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
            elif observation == "cameras":
                camera_obs = np.stack((robot_state.camera_60,
                                      robot_state.camera_180,
                                      robot_state.camera_300), axis=0)
                observations_dict["cameras"] = camera_obs
            elif observation in self.observation_functions:
                observations_dict[observation] = \
                    self.observation_functions[observation](robot_state)
            else:
                raise Exception("The robot doesn't know about observation "
                                "key {}".format(observation))
        #now normalize everything here
        if self.normalized_observations:
            for key in observations_dict.keys():
                observations_dict[key] = \
                    self.normalize_observation_for_key(observations_dict[key],
                                                       key)

        return observations_dict

    def get_current_camera_observations(self, robot_state):
        camera_obs = np.stack((robot_state.camera_60,
                               robot_state.camera_180,
                               robot_state.camera_300), axis=0)
        if self.normalized_observations:
            camera_obs = self.normalize_observation_for_key(camera_obs,
                                                            "cameras")
        return camera_obs




