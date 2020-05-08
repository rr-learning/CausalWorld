from scipy.spatial.transform import Rotation as R


def euler_to_quaternion(euler_angles):
    r = R.from_rotvec(euler_angles)
    return r.as_quat()
