import numpy as np
import math

from gym import spaces


class TriFingerAction(object):
    def __init__(self, action_mode, normalize_actions=True):
        pass

    def set_action_space(self):
        raise Exception(" Not implemented")

    def get_action_space(self):
        raise Exception(" Not implemented")

    def is_normalized(self):
        raise Exception(" Not implemented")

    def normalize_actions(self):
        raise Exception(" Not implemented")

    def satisfy_constraints(self):
        raise Exception(" Not implemented")

    def clip_actions(self):
        raise Exception(" Not implemented")

    def normalize_action(self):
        raise Exception(" Not implemented")

    def denormalize_action(self):
        raise Exception(" Not implemented")


