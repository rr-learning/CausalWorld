import numpy as np


def calculate_end_effector_to_goal(end_effector_position, goal_position):
    """

    :param end_effector_position:
    :param goal_position:
    :return:
    """
    flat_goals = np.concatenate([goal_position] * 3)
    end_effector_to_goal = list(np.subtract(flat_goals, end_effector_position))
    return end_effector_to_goal


def get_suggested_grip_locations(cuboid_size, cuboid_rotation_matrix_w_c):
    """

    :param cuboid_size:
    :param cuboid_rotation_matrix_w_c:
    :return:
    """
    grip_locations = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0],
                      [0, 0, 0.5], [0, 0, -0.5]]
    face_normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                    [0, 0, -1]]
    grip_locations = np.array(grip_locations) * (cuboid_size + 0.0035)
    grip_locations = np.concatenate([grip_locations, np.ones([6, 1])], axis=1)
    grip_locations_rotated = np.matmul(cuboid_rotation_matrix_w_c,
                                       np.transpose(grip_locations))
    rotated_face_normals = np.matmul(cuboid_rotation_matrix_w_c[:3, :3],
                                     np.transpose(face_normals))
    np.argmax(rotated_face_normals, axis=0)
    grasp_index_red = np.argmax(rotated_face_normals,
                                axis=1)[1]
    if grasp_index_red % 2 == 0:
        grasp_index_green = grasp_index_red + 1
    else:
        grasp_index_green = grasp_index_red - 1
    return np.transpose(
        grip_locations_rotated)[[grasp_index_red, grasp_index_green], :3]


def combine_intervention_spaces(cont_bound_a, cont_bound_b):
    """

    :param cont_bound_a:
    :param cont_bound_b:
    :return:
    """
    if not isinstance(cont_bound_b[0], list) and not isinstance(
            cont_bound_b[0], np.ndarray):
        if cont_bound_a[0] < cont_bound_b[0]:
            return np.array([cont_bound_a[0], cont_bound_b[1]])
        else:
            return np.array([cont_bound_b[0], cont_bound_a[1]])
    else:
        lb = []
        ub = []
        for val_ind in range(len(cont_bound_b[0])):
            if cont_bound_a[0][val_ind] < cont_bound_b[0][val_ind]:
                lb.append(cont_bound_a[0][val_ind])
                ub.append(cont_bound_b[1][val_ind])
            elif cont_bound_a[0][val_ind] == cont_bound_b[0][val_ind]:
                lb.append(cont_bound_a[0][val_ind])
                if cont_bound_a[1][val_ind] < cont_bound_b[1][val_ind]:
                    ub.append(cont_bound_b[1][val_ind])
                else:
                    ub.append(cont_bound_a[1][val_ind])
            else:
                lb.append(cont_bound_b[0][val_ind])
                ub.append(cont_bound_a[1][val_ind])
        lb = np.array(lb)
        ub = np.array(ub)
        return np.array([lb, ub])
