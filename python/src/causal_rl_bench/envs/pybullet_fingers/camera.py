import pybullet
import numpy as np


class Camera(object):
    def __init__(self, camera_position, camera_orientation, pybullet_client):
        self._translation = camera_position
        self._camera_orientation = camera_orientation
        self._pybullet_client = pybullet_client
        self._width = 72
        self._height = 54
        x = self._camera_orientation[0]
        y = self._camera_orientation[1]
        z = self._camera_orientation[2]
        w = self._camera_orientation[3]
        self._view_matrix = self._pybullet_client.computeViewMatrix(
            cameraEyePosition=self._translation,
            cameraTargetPosition=[2 * (x * z + w * y),
                                  2 * (y * z - w * x),
                                  1 - 2 * (x * x + y * y)],
            cameraUpVector=[2 * (x * y - w * z),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z + w * x)])

        self._proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
                            fov=52, aspect=float(self._width) / self._height,
                            nearVal=0.001, farVal=100.0)

    def get_image(self):
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
                            width=self._width, height=self._height,
                            viewMatrix=self._view_matrix,
                            projectionMatrix=self._proj_matrix,
                            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
