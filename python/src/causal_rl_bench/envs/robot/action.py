import numpy as np
import math

from gym import spaces


class TriFingerActionSpace:
    def __init__(self, action_mode="joint_positions"):
        self.num_fingers = 3
        self.max_motor_torque = 0.36
        if action_mode == "joint_positions":
            self.action_bounds = {
                "low": np.array(
                    [-math.radians(70), -math.radians(70), -math.radians(160)]
                    * self.num_fingers
                ),
                "high": np.array(
                    [math.radians(70), 0, math.radians(-2)] * self.num_fingers
                ),
            }
        elif action_mode == "torques":
            self.action_bounds = {
                "low": np.array(
                    [-self.max_motor_torque] * 3 * self.num_fingers
                ),
                "high": np.array(
                    [self.max_motor_torque] * 3 * self.num_fingers
                ),
            }
        elif action_mode == "both":
            self.action_bounds = {
                "low": np.array(
                    [-self.max_motor_torque] * 3 * self.num_fingers + [-math.radians(70),
                                                                       -math.radians(70),
                                                                       -math.radians(160)] * self.num_fingers
                ),
                "high": np.array(
                    [self.max_motor_torque] * 3 * self.num_fingers + [math.radians(70),
                                                                      0,
                                                                      math.radians(-2)] * self.num_fingers
                ),
            }
        else:
            raise ValueError("No valid action_mode specified: {}".format(action_mode))

    def get_unscaled_action_space(self):
        """
        Returns the unscaled action space according to the action bounds.
        """
        return spaces.Box(
            low=self.action_bounds["low"],
            high=self.action_bounds["high"],
            dtype=np.float64,
        )

    def get_scaled_action_space(self):
        """
        Returns an action space with the same size as the unscaled
        but bounded by -1s and 1s.
        """
        return spaces.Box(
            low=-np.ones(len(self.action_bounds["low"])),
            high=np.ones(len(self.action_bounds["high"])),
            dtype=np.float64
            )
