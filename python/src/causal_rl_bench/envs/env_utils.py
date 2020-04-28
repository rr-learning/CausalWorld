import math
import numpy as np


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


