import pybullet
import numpy as np


class SilhouetteObject(object):
    def __init__(self, name):
        self.name = name

    def get_state(self):
        raise Exception("get_state not implemented")

    def get_bounds(self):
        raise Exception("get_bounds not implemented")


class SCuboid(SilhouetteObject):
    def __init__(
        self,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        alpha=0.5, colour="red"
    ):
        super(SCuboid, self).__init__(name)
        self.type_id = 0
        self.size = size
        if colour == "red":
            self.colour = 0
            rgbaColor = [1, 0, 0, alpha]
        elif colour == "blue":
            self.colour = 1
            rgbaColor = [0, 0, 1, alpha]
        elif colour == "green":
            self.colour = 3
            rgbaColor = [0, 1, 0, alpha]
        else:
            raise Exception("The colour specified is not supported")
        self.block_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size)/2,
            rgbaColor=rgbaColor
        )
        self.block = pybullet.createMultiBody(
            baseVisualShapeIndex=self.block_id,
            basePosition=position,
            baseOrientation=orientation
        )
        self.state = dict()
        self.state["silhouette_" + self.name + "_type"] = self.type_id
        self.state["silhouette_" + self.name + "_position"] = position
        self.state["silhouette_" + self.name + "_orientation"] = orientation
        self.state["silhouette_" + self.name + "_size"] = self.size
        self.state["silhouette_" + self.name + "_colour"] = self.colour

        # specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds["silhouette_" + self.name + "_type"] = 0
        self.lower_bounds["silhouette_" + self.name + "_position"] = \
            np.array([-0.5] * 3 * 3)
        self.lower_bounds["silhouette_" + self.name + "_orientation"] = \
            np.array([-10] * 4 * 3)
        self.lower_bounds["silhouette_" + self.name + "_size"] = \
            np.array([0.065, 0.065, 0.065])
        self.lower_bounds["silhouette_" + self.name + "_colour"] = \
            np.array([0])

        self.upper_bounds["silhouette_" + self.name + "_type"] = 10
        self.upper_bounds["silhouette_" + self.name + "_position"] = \
            np.array([0.5] * 3 * 3)
        self.upper_bounds["silhouette_" + self.name + "_orientation"] = \
            np.array([10] * 4 * 3)
        self.upper_bounds["silhouette_" + self.name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.upper_bounds["silhouette_" + self.name + "_colour"] = \
            np.array([2])

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
        return self.state

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

