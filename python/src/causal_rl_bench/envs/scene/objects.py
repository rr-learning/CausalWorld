import pybullet
import numpy as np


class RigidObject(object):
    def __init__(self, name):
        self.name = name

    def get_state(self):
        raise Exception("get_state not implemented")

    def get_bounds(self):
        raise Exception("get_bounds not implemented")


class Cuboid(RigidObject):
    def __init__(
        self,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        mass=0.08, colour="red"
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
        super(Cuboid, self).__init__(name)
        self.type_id = 0
        self.mass = mass
        self.size = size
        self.not_fixed = True
        if colour == "red":
            self.colour = 0
            rgbaColor = [1, 0, 0, 1]
        elif colour == "blue":
            self.colour = 1
            rgbaColor = [0, 0, 1, 1]
        elif colour == "green":
            self.colour = 2
            rgbaColor = [0, 1, 0, 1]
        else:
            raise Exception("The colour specified is not supported")
        self.block_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, halfExtents=np.array(size)/2)
        self.block = pybullet.createMultiBody(
            baseCollisionShapeIndex=self.block_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=mass
        )
        pybullet.changeVisualShape(self.block, -1, rgbaColor=rgbaColor)
        #specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds["rigid_" + self.name + "_type"] = 0
        self.lower_bounds["rigid_" + self.name + "_position"] = \
            np.array([-0.5] * 3 * 3)
        self.lower_bounds["rigid_" + self.name + "_orientation"] = \
            np.array([-10] * 4 * 3)
        self.lower_bounds["rigid_" + self.name + "_velocity"] = \
            np.array([-0.5] * 3 * 3)
        self.lower_bounds["rigid_" + self.name + "_mass"] = \
            np.array([0])
        self.lower_bounds["rigid_" + self.name + "_size"] = \
            np.array([0.065, 0.065, 0.065])
        self.lower_bounds["rigid_" + self.name + "_colour"] = \
            np.array([0])

        self.upper_bounds["rigid_" + self.name + "_type"] = 10
        self.upper_bounds["rigid_" + self.name + "_position"] = \
            np.array([0.5] * 3 * 3)
        self.upper_bounds["rigid_" + self.name + "_orientation"] = \
            np.array([10] * 4 * 3)
        self.upper_bounds["rigid_" + self.name + "_velocity"] = \
            np.array([0.5] * 3 * 3)
        self.upper_bounds["rigid_" + self.name + "_mass"] = \
            np.array([0.2])
        self.upper_bounds["rigid_" + self.name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.upper_bounds["rigid_" + self.name + "_colour"] = \
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
        state = dict()
        position, orientation = pybullet.getBasePositionAndOrientation(
            self.block
        )
        state["rigid_" + self.name + "_type"] = self.type_id
        state["rigid_" + self.name + "_position"] = position
        state["rigid_" + self.name + "_orientation"] = orientation
        velocity = pybullet.getBaseVelocity(self.block)
        state["rigid_" + self.name + "_velocity"] = velocity
        state["rigid_" + self.name + "_mass"] = self.mass
        state["rigid_" + self.name + "_size"] = self.size
        state["rigid_" + self.name + "_colour"] = self.colour
        return state

    def is_not_fixed(self):
        return self.not_fixed

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds
