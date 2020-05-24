import numpy as np


def calculate_end_effector_to_goal(end_effector_position, goal_position):
    flat_goals = np.concatenate([goal_position] * 3)
    end_effector_to_goal = list(
        np.subtract(flat_goals, end_effector_position)
    )
    return end_effector_to_goal


def get_suggested_grip_locations(cuboid_size, cuboid_rotation_matrix_w_c):
    grip_locations = [[0.5, 0, 0],
                      [-0.5, 0, 0],
                      [0, 0.5, 0],
                      [0, -0.5, 0],
                      [0, 0, 0.5],
                      [0, 0, -0.5]]
    face_normals = [[1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1]]
    grip_locations = np.array(grip_locations) * (cuboid_size + 0.01)
    grip_locations = np.concatenate([grip_locations, np.ones([6, 1])], axis=1)
    # face_normals = np.concatenate([face_normals, np.ones([6, 1])], axis=1)
    grip_locations_rotated = np.matmul(cuboid_rotation_matrix_w_c, np.transpose(grip_locations))
    rotated_face_normals = np.matmul(cuboid_rotation_matrix_w_c[:3, :3], np.transpose(face_normals))
    np.argmax(rotated_face_normals, axis=0)
    grasp_index_red = np.argmax(rotated_face_normals, axis=1)[1] #TODO: remove this heuristic
    if grasp_index_red % 2 == 0:
        grasp_index_green = grasp_index_red + 1
    else:
        grasp_index_green = grasp_index_red - 1
    #red finger operates in the upper half
    return np.transpose(grip_locations_rotated)[[grasp_index_red, grasp_index_green], :3]
