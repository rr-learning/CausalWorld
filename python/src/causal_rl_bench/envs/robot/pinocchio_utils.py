import numpy as np

import pinocchio
from causal_rl_bench.configs.world_constants import WorldConstants


class PinocchioUtils:
    """
    Consists of kinematic methods for the finger platform.
    """

    def __init__(self, finger_urdf_path, tip_link_names):
        """
        Initializes the finger model on which control's to be performed.

        Args:
            finger (SimFinger): An instance of the SimFinger class
        """
        self.robot_model = pinocchio.buildModelFromUrdf(finger_urdf_path)
        self.data = self.robot_model.createData()
        self.tip_link_ids = [
            self.robot_model.getFrameId(link_name)
            for link_name in tip_link_names
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
            self.robot_model, self.data, joint_positions,
        )
        result = [
            np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist()
            for link_id in self.tip_link_ids
        ]
        result = np.concatenate(result)
        result[2] -= WorldConstants.FLOOR_HEIGHT
        result[5] -= WorldConstants.FLOOR_HEIGHT
        result[8] -= WorldConstants.FLOOR_HEIGHT
        return result
