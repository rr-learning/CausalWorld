import pybullet
import numpy as np


class Camera(object):

    def __init__(self, camera_position, camera_orientation, pybullet_client_id):
        self._translation = camera_position
        self._camera_orientation = camera_orientation
        self._pybullet_client_id = pybullet_client_id
        self._width = 128
        self._height = 128
        x = self._camera_orientation[0]
        y = self._camera_orientation[1]
        z = self._camera_orientation[2]
        w = self._camera_orientation[3]
        self._view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=self._translation,
            cameraTargetPosition=[
                2 * (x * z + w * y), 2 * (y * z - w * x),
                1 - 2 * (x * x + y * y)
            ],
            cameraUpVector=[
                2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                2 * (y * z + w * x)
            ],
            physicsClientId=self._pybullet_client_id)

        self._proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=52,
            aspect=float(self._width) / self._height,
            nearVal=0.001,
            farVal=100.0,
            physicsClientId=self._pybullet_client_id)

    def get_image(self):
        (_, _, px, _, _) = pybullet.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._pybullet_client_id)
        rgb_array = np.array(px)
        if rgb_array.ndim == 1:
            rgb_array = rgb_array.reshape((self._width, self._height, 4))
        rgb_array = rgb_array[::-1, :, :3]
        return rgb_array
