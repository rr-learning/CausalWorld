"""
This file should contain all the constants related to the trifinger model
itself (i.e: the urdf).
"""
import numpy as np


class WorldConstants:
    ROBOT_ID = 1
    FLOOR_ID = 2
    STAGE_ID = 3
    FLOOR_HEIGHT = 0.011
    ROBOT_HEIGHT = 0.34
    ARENA_BB = np.array([[-0.15, -0.15, 0], [0.15, 0.15, 0.3]])
    LINK_IDS = {
        'robot_finger_60_link_0': 1,
        'robot_finger_60_link_1': 2,
        'robot_finger_60_link_2': 3,
        'robot_finger_60_link_3': 4,
        'robot_finger_120_link_0': 6,
        'robot_finger_120_link_1': 7,
        'robot_finger_120_link_2': 8,
        'robot_finger_120_link_3': 9,
        'robot_finger_300_link_0': 11,
        'robot_finger_300_link_1': 12,
        'robot_finger_300_link_2': 13,
        'robot_finger_300_link_3': 14
    }

    VISUAL_SHAPE_IDS = {
        'robot_finger_60_link_0': 0,
        'robot_finger_60_link_1': 4,
        'robot_finger_60_link_2': 5,
        'robot_finger_60_link_3': 6,
        'robot_finger_120_link_0': 7,
        'robot_finger_120_link_1': 11,
        'robot_finger_120_link_2': 12,
        'robot_finger_120_link_3': 13,
        'robot_finger_300_link_0': 14,
        'robot_finger_300_link_1': 15,
        'robot_finger_300_link_2': 16,
        'robot_finger_300_link_3': 17
    }

    JOINT_NAMES = [
        "finger_upper_link_0",
        "finger_middle_link_0",
        "finger_lower_link_0",
        "finger_upper_link_120",
        "finger_middle_link_120",
        "finger_lower_link_120",
        "finger_upper_link_240",
        "finger_middle_link_240",
        "finger_lower_link_240",
    ]
    TIP_LINK_NAMES = [
        "finger_tip_link_0",
        "finger_tip_link_120",
        "finger_tip_link_240",
    ]
