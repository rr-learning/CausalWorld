import numpy as np
import math

from gym import spaces


class TriFingerAction(object):
    def __init__(self, action_mode="joint_positions", normalize_actions=True):
        """
        This class is responsible for the robot action limits and its spaces.

        :param action_mode:
        :param normalize_actions:
        """
        self.normalize_actions = normalize_actions
        self.max_motor_torque = 0.36
        self.low = None
        self.high = None
        num_fingers = 3
        self.action_mode = action_mode
        self.joint_positions_lower_bounds = np.array([-math.radians(70),
                                                      -math.radians(70),
                                                      -math.radians(160)] * 3)
        self.joint_positions_upper_bounds = np.array([math.radians(70),
                                                      0,
                                                      math.radians(-2)] * 3)
        if action_mode == "joint_positions":
            lower_bounds = self.joint_positions_lower_bounds
            upper_bounds = self.joint_positions_upper_bounds

        elif action_mode == "joint_torques":
            lower_bounds = np.array([-self.max_motor_torque] * 3 * num_fingers)
            upper_bounds = np.array([self.max_motor_torque] * 3 * num_fingers)

        elif action_mode == "end_effector_positions":
            lower_bounds = np.array([-0.5, -0.5, 0] * 3)
            upper_bounds = np.array([0.5, 0.5, 0.5] * 3)

        else:
            raise ValueError("No valid action_mode specified: {}".
                             format(action_mode))
        self.set_action_space(lower_bounds, upper_bounds)

    def set_action_space(self, lower_bounds, upper_bounds):
        """

        :param lower_bounds:
        :param upper_bounds:

        :return:
        """
        assert len(lower_bounds) == len(upper_bounds)
        self.low = lower_bounds
        self.high = upper_bounds

    def get_action_space(self):
        """

        :return:
        """
        if self.normalize_actions:
            return spaces.Box(low=-np.ones(len(self.low)),
                              high=np.ones(len(self.high)),
                              dtype=np.float64)
        else:
            return spaces.Box(low=self.low,
                              high=self.high,
                              dtype=np.float64)

    def is_normalized(self):
        """

        :return:
        """
        return self.normalize_actions

    def satisfy_constraints(self, action):
        """

        :param action:

        :return:
        """
        if self.normalize_actions:
            return (action > -1.).all() and (action < 1.).all()
        else:
            return (action > self.low).all() and (action < self.high).all()

    def clip_action(self, action):
        """

        :param action:

        :return:
        """
        if self.normalize_actions:
            return np.clip(action, -1.0, 1.0)
        else:
            return np.clip(action, self.low, self.high)

    def normalize_action(self, action):
        """

        :param action:

        :return:
        """
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0

    def denormalize_action(self, action):
        """

        :param action:

        :return:
        """
        return self.low + (action + 1.0) / 2.0 * \
               (self.high - self.low)
