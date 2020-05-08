import numpy as np


def calculate_end_effector_to_goal(end_effector_position, goal_position):
    flat_goals = np.concatenate([goal_position] * 3)
    end_effector_to_goal = list(
        np.subtract(flat_goals, end_effector_position)
    )
    return end_effector_to_goal