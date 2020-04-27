import numpy as np
import math

from gym import spaces


class TriFingerAction(object):
    def __init__(self, action_mode, normalize_actions=True):
        self.normalized_actions = normalize_actions
        self.num_fingers = 3
        self.max_motor_torque = 0.36

        if action_mode == "joint_positions":
            lower_bounds = np.array([-math.radians(70), -math.radians(70), -math.radians(160)] * self.num_fingers)
            upper_bounds = np.array([math.radians(70), 0, math.radians(-2)] * self.num_fingers)
        elif action_mode == "torques":
            lower_bounds = np.array([-self.max_motor_torque] * 3 * self.num_fingers)
            upper_bounds = np.array([self.max_motor_torque] * 3 * self.num_fingers)
        elif action_mode == "both":
            lower_bounds = np.array([-self.max_motor_torque] * 3 * self.num_fingers
                                + [-math.radians(70), -math.radians(70), -math.radians(160)] * self.num_fingers)
            upper_bounds = np.array([self.max_motor_torque] * 3 * self.num_fingers
                                 + [math.radians(70), 0, math.radians(-2)] * self.num_fingers)
        else:
            raise ValueError("No valid action_mode specified: {}".format(action_mode))

        self.set_action_space(lower_bounds, upper_bounds)

    def set_action_space(self, lower_bounds, upper_bounds):
        assert len(lower_bounds) == len(upper_bounds)
        self.low = lower_bounds
        self.high = upper_bounds

    def get_action_space(self):
        if self.normalized_actions:
            return spaces.Box(low=-np.ones(len(self.low)),
                              high=np.ones(len(self.high)),
                              dtype=np.float64)
        else:
            return spaces.Box(low=self.low,
                              high=self.high,
                              dtype=np.float64)

    def is_normalized(self):
        return self.normalized_actions

    def satisfy_constraints(self, action):
        if self.normalized_actions:
            return (action > -1.).all() and (action < 1.).all()
        else:
            low = self.low
            high = self.high
            return (action > low).all() and (action < high).all()

    def clip_action(self, action):
        if self.normalized_actions:
            return np.clip(action, -1.0, 1.0)
        else:
            low = self.low
            high = self.high
            return np.clip(action, low, high)

    def normalize_action(self, action):
        low = self.low
        high = self.high
        return 2.0 * (action - low) / (high - low) - 1.0

    def denormalize_action(self, action):
        low = self.low
        high = self.high
        return low + (action + 1.0) / 2.0 * (high - low)



