import numpy as np
import math

from gym import spaces


class TriFingerAction(object):
    def __init__(self, action_mode, normalize_actions=True):
        self.normalize_actions = normalize_actions
        self.max_motor_torque = 0.36
        self.low = None
        self.high = None
        num_fingers = 3
        self.action_mode = action_mode
        if action_mode == "joint_positions":
            lower_bounds = np.array([-math.radians(70), -math.radians(70),
                                     -math.radians(160)] * num_fingers)
            upper_bounds = np.array([math.radians(70), 0, math.radians(-2)]
                                    * num_fingers)
        elif action_mode == "joint_torques":
            lower_bounds = np.array([-self.max_motor_torque] * 3 * num_fingers)
            upper_bounds = np.array([self.max_motor_torque] * 3 * num_fingers)
        else:
            raise ValueError("No valid action_mode specified: {}".
                             format(action_mode))
        self.set_action_space(lower_bounds, upper_bounds)

    def set_action_space(self, lower_bounds, upper_bounds):
        assert len(lower_bounds) == len(upper_bounds)
        self.low = lower_bounds
        self.high = upper_bounds

    def get_action_space(self):
        if self.normalize_actions:
            return spaces.Box(low=-np.ones(len(self.low)),
                              high=np.ones(len(self.high)),
                              dtype=np.float64)
        else:
            return spaces.Box(low=self.low,
                              high=self.high,
                              dtype=np.float64)

    def is_normalized(self):
        return self.normalize_actions

    def satisfy_constraints(self, action):
        if self.normalize_actions:
            return (action > -1.).all() and (action < 1.).all()
        else:
            return (action > self.low).all() and (action < self.high).all()

    def clip_action(self, action):
        if self.normalize_actions:
            return np.clip(action, -1.0, 1.0)
        else:
            return np.clip(action, self.low, self.high)

    def normalize_action(self, action):
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0

    def denormalize_action(self, action):
        return self.low + (action + 1.0) / 2.0 * (self.high - self.low)

    def sample_actions(self, sampling_strategy="uniform", mode=None):
        #TODO: why are the numbers different from the bounds?
        if mode is None:
            mode = self.action_mode
        if mode == "joint_positions" and \
                sampling_strategy == "uniform":
            list_to_return = []
            upper = np.random.uniform(-math.radians(30), math.radians(30))
            middle = np.random.uniform(-math.radians(60), math.radians(60))
            lower = np.random.uniform(-math.radians(100), -math.radians(2))
            list_to_return += [upper, middle, lower]
            return list_to_return
        else:
            raise Exception("not yet implemented")
