#!/usr/bin/env python3
import numpy as np


class Action:
    """
    Create the action data structure used by the SimFinger class.
    """

    def __init__(self, t, p, kp=None, kd=None):
        self.torque = t
        self.position = p

        if kp is None:
            self.position_kp = np.full_like(p, np.nan, dtype=float)
        else:
            self.position_kp = kp

        if kd is None:
            self.position_kd = np.full_like(p, np.nan, dtype=float)
        else:
            self.position_kd = kd
