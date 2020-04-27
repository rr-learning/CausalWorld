import pybullet
import numpy as np


class Cuboid:
    def __init__(
        self,
        size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        mass=0.08,
    ):
        """
        Import the block

        Args:
            position (list): where in xyz space should the block
                be imported
            orientation (list): initial orientation quaternion of the block
            half_size (float): how large should this block be
            mass (float): how heavy should this block be
        """
        self.not_fixed = True
        self.block_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, halfExtents=np.array(size)/2
        )
        self.block = pybullet.createMultiBody(
            baseCollisionShapeIndex=self.block_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=mass
        )

    def set_state(self, position, orientation):
        """
        Resets the block state to the provided position and
        orientation

        Args:
            position: the position to which the block is to be
                set
            orientation: desired to be set
        """
        pybullet.resetBasePositionAndOrientation(
            self.block, position, orientation
        )

    def get_state(self):
        """
        Returns:
            Current position and orientation of the block.
        """
        position, orientation = pybullet.getBasePositionAndOrientation(
            self.block
        )
        return list(position), list(orientation)

    def is_not_fixed(self):
        return self.not_fixed
