import numpy as np

import pinocchio
from causal_rl_bench.configs.world_constants import WorldConstants


class PinocchioUtils(object):
    def __init__(self, finger_urdf_path):
        self._robot_model = pinocchio.buildModelFromUrdf(finger_urdf_path)
        self._data = self._robot_model.createData()
        self._tip_link_ids = [
            self._robot_model.getFrameId(link_name)
            for link_name in WorldConstants.TIP_LINK_NAMES
        ]

    def forward_kinematics(self, joint_positions):
        """
        Compute end effector positions for the given joint configuration.
        Args:
            finger (SimFinger): a SimFinger object
            joint_positions (list): Flat list of angular joint positions.
        Returns:
            List of end-effector positions. Each position is given as an
            np.array with x,y,z positions.
        """
        pinocchio.framesForwardKinematics(
            self._robot_model, self._data, joint_positions,
        )

        return [
            np.asarray(self._data.oMf[link_id].translation).reshape(-1).tolist()
            for link_id in self._tip_link_ids
        ]