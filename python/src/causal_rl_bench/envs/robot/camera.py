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
            cameraTargetPosition=[0, 0, 0.01],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._pybullet_client_id)

        self._proj_matrix = pybullet.computeProjectionMatrixFOV(
                            fov=51, aspect=float(self._width) / self._height,
                            nearVal=0.002, farVal=1,
                            physicsClientId=self._pybullet_client_id)

    def get_image(self):
        (_, _, px, _, _) = pybullet.getCameraImage(
                            width=self._width, height=self._height,
                            viewMatrix=self._view_matrix,
                            projectionMatrix=self._proj_matrix,
                            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                            physicsClientId=self._pybullet_client_id)
        rgb_array = np.array(px)
        rgb_array = rgb_array[::-1, :, :3]
        return rgb_array
