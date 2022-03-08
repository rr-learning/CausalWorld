import numpy as np
from gym import spaces

try:
    # numpy versions 1.17+ have np.core.umath.clip
    # as faster alternative to 1.17+ np.clip
    clip = np.core.umath.clip
except AttributeError:
    clip = np.clip

def scale(x, space):
    """

    :param x:
    :param space:
    :return:
    """
    return 2.0 * (x - space.low) / (space.high - space.low) - 1.0


def unscale(y, space):
    """

    :param y:
    :param space:
    :return:
    """
    return space.low + (y + 1.0) / 2.0 * (space.high - space.low)


def combine_spaces(space_1, space_2):
    """

    :param space_1:
    :param space_2:
    :return:
    """
    lower_bound = np.concatenate((space_1.low, space_2.low))
    upper_bound = np.concatenate((space_1.high, space_2.high))
    return spaces.Box(low=lower_bound, high=upper_bound, dtype=np.float64)
