import numpy as np

from gym import spaces


class TriFingerAction(object):

    def __init__(self, action_mode="joint_positions", normalize_actions=True):
        """
        This class is responsible for the robot action limits and its spaces.

        :param action_mode: (str) this can be "joint_positions", "joint_torques" or
                                  "end_effector_positions".
        :param normalize_actions: (bool) true if actions should be normalized.
        """
        self.normalize_actions = normalize_actions
        self.max_motor_torque = 0.36
        self.low = None
        self.high = None
        num_fingers = 3
        self.action_mode = action_mode
        self.joint_positions_lower_bounds = np.array(
            [-1.57, -1.2, -3.0] * 3)
        self.joint_positions_upper_bounds = np.array(
            [1.0, 1.57, 3.0] * 3)

        self.joint_positions_raised = np.array([-1.56, -0.08, -2.7] * 3)
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
            raise ValueError(
                "No valid action_mode specified: {}".format(action_mode))
        self.set_action_space(lower_bounds, upper_bounds)

    def set_action_space(self, lower_bounds, upper_bounds):
        """

        :param lower_bounds: (list) array of the lower bounds of actions.
        :param upper_bounds: (list) array of the upper bounds of actions.

        :return:
        """
        assert len(lower_bounds) == len(upper_bounds)
        self.low = lower_bounds
        self.high = upper_bounds

    def get_action_space(self):
        """

        :return: (gym.spaces.Box) returns the current actions space.
        """
        if self.normalize_actions:
            return spaces.Box(low=-np.ones(len(self.low)),
                              high=np.ones(len(self.high)),
                              dtype=np.float64)
        else:
            return spaces.Box(low=self.low, high=self.high, dtype=np.float64)

    def is_normalized(self):
        """

        :return: (bool) returns true if actions are normalized, false otherwise.
        """
        return self.normalize_actions

    def satisfy_constraints(self, action):
        """

        :param action: (nd.array) action to check if it satisfies the constraints.

        :return: (bool) returns true if the action satisfies all constraints.
        """
        if self.normalize_actions:
            return (action > -1.).all() and (action < 1.).all()
        else:
            return (action > self.low).all() and (action < self.high).all()

    def clip_action(self, action):
        """

        :param action: (nd.array) action to clip to the limits.

        :return: (nd.array) clipped action.
        """
        if self.normalize_actions:
            return np.clip(action, -1.0, 1.0)
        else:
            return np.clip(action, self.low, self.high)

    def normalize_action(self, action):
        """

        :param action: (nd.array) action to normalize.

        :return: (nd.array) normalized action.
        """
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0

    def denormalize_action(self, action):
        """

        :param action: (nd.array) action to denormalize.

        :return: (nd.array) denormalized action.
        """
        return self.low + (action + 1.0) / 2.0 * \
               (self.high - self.low)
