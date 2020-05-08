import math
import numpy as np
from gym import spaces

def scale(x, space):
    """
    Scale some input to be between the range [-1;1] from the range
    of the space it belongs to
    """
    return 2.0 * (x - space.low) / (space.high - space.low) - 1.0


def unscale(y, space):
    """
    Unscale some input from [-1;1] to the range of another space
    """
    return space.low + (y + 1.0) / 2.0 * (space.high - space.low)


def combine_spaces(space_1, space_2):
    lower_bound = np.array([])
    lower_bound = np.append(lower_bound, space_1.low)
    lower_bound = np.append(lower_bound, space_2.low)
    upper_bound = np.array([])
    upper_bound = np.append(upper_bound, space_1.high)
    upper_bound = np.append(upper_bound, space_2.high)
    return spaces.Box(low=lower_bound,
                      high=upper_bound,
                      dtype=np.float64)

