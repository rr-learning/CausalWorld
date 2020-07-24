from scipy.spatial.transform import Rotation as R
import numpy as np


def euler_to_quaternion(euler_angles):
    r = R.from_rotvec(euler_angles)
    return r.as_quat()


def quaternion_to_euler(quat_angles):
    r = R.from_quat(quat_angles)
    return r.as_rotvec()


def quaternion_conjugate(quaternion):
    inv_q = -quaternion
    inv_q[:, 3] *= -1
    return inv_q


def cart2cyl(position):
    x = position[0]
    y = position[1]
    z = position[2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi, z])


def cyl2cart(position):
    x = position[0] * np.cos(position[1])
    y = position[0] * np.sin(position[1])
    z = position[2]
    return np.array([x, y, z])


def quaternion_mul(q0, q1):
    x0 = q0[:, 0]
    y0 = q0[:, 1]
    z0 = q0[:, 2]
    w0 = q0[:, 3]

    x1 = q1[:, 0]
    y1 = q1[:, 1]
    z1 = q1[:, 2]
    w1 = q1[:, 3]
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q = np.array([x, y, z, w])
    q = q.swapaxes(0, 1)
    return q


def rotate_points(points_batch, r_quaternion):
    r = R.from_quat(r_quaternion)
    return np.transpose(np.matmul(r.as_matrix(), np.transpose(points_batch)))


def get_transformation_matrix(translation, r_quaternion):
    r = R.from_quat(r_quaternion)
    r_matrix = np.zeros([4, 4])
    r_matrix[:3, :3] = r.as_matrix()
    r_matrix[:3, 3] = translation
    r_matrix[3, 3] = 1
    return r_matrix


def get_rotation_matrix(r_quaternion):
    r = R.from_quat(r_quaternion)
    return r.as_matrix()
