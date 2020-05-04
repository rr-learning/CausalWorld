import pybullet
import numpy as np


class SilhouetteObject(object):
    def __init__(self, name):
        self.name = name

    def get_state(self, state_type='dict'):
        raise Exception("get_state not implemented")

    def set_state(self, state_dict):
        raise Exception("set_state not implemented")

    def set_full_state(self, new_state):
        raise Exception("set_state not implemented")

    def get_bounds(self):
        raise Exception("get_bounds not implemented")

    def do_intervention(self, variable_name, variable_value):
        raise Exception("do_intervention not implemented")

    def get_state_variable_names(self):
        raise Exception("get_state_variable_names not implemented")

    def get_state_size(self):
        raise Exception("get_state_size not implemented")

    def set_pose(self, position, orientation):
        raise Exception("get_state_size not implemented")


class SCuboid(SilhouetteObject):
    def __init__(
        self,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        alpha=0.2, colour=np.array([1, 0, 0])
    ):
        super(SCuboid, self).__init__(name)
        self.type_id = 0
        self.size = size
        self.colour = colour
        self.alpha = alpha
        self.block_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size)/2,
            rgbaColor=np.append(self.colour, alpha)
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
        self.lower_bounds["silhouette_" + self.name + "_type"] = np.array([0])
        self.lower_bounds["silhouette_" + self.name + "_position"] = \
            np.array([-0.5] * 3)
        self.lower_bounds["silhouette_" + self.name + "_orientation"] = \
            np.array([-10] * 4)
        self.lower_bounds["silhouette_" + self.name + "_size"] = \
            np.array([0.065, 0.065, 0.065])
        self.lower_bounds["silhouette_" + self.name + "_colour"] = \
            np.array([0]*3)

        #TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds["silhouette_" + self.name + "_type"] = np.array([10])
        self.upper_bounds["silhouette_" + self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds["silhouette_" + self.name + "_orientation"] = \
            np.array([10] * 4)
        self.upper_bounds["silhouette_" + self.name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.upper_bounds["silhouette_" + self.name + "_colour"] = \
            np.array([1]*3)

        self._state_variable_names = ['type', 'position',
                                      'orientation', 'size', 'colour']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds["silhouette_" + self.name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]

    def set_full_state(self, new_state):
        #form dict first
        new_state_dict = dict()
        start = 0
        for i in range(len(self._state_variable_sizes)):
            end = start + self._state_variable_sizes[i]
            new_state_dict[self._state_variable_names[i]] = new_state[start:end]
            start = end
        self.set_state(new_state_dict)
        return

    def set_state(self, state_dict):
        if 'position' not in state_dict or 'orientation' not in state_dict:
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'size' in state_dict:
            pybullet.removeBody(self.block)
            self.block_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(state_dict['size']) / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block = pybullet.createMultiBody(
                baseVisualShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation
            )
        elif 'position' in state_dict or 'orientation' in state_dict:
            pybullet.resetBasePositionAndOrientation(
                self.block, position, orientation
            )
        if 'colour' in  state_dict:
            pybullet.changeVisualShape(self.block, -1,
                                       rgbaColor=
                                       np.append(state_dict['colour'], self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        #TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
            pybullet.resetBasePositionAndOrientation(
                self.block, variable_value, orientation
            )
        elif variable_name == 'orientation':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
            pybullet.resetBasePositionAndOrientation(
                self.block, position, variable_value
            )
        elif variable_name == 'size':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
            pybullet.removeBody(self.block)
            self.block_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(variable_value) / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block = pybullet.createMultiBody(
                baseVisualShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation
            )
            pybullet.changeVisualShape(self.block, -1,
                                       rgbaColor=np.append(self.colour,
                                                           self.alpha))
        elif variable_name == 'colour':
            pybullet.changeVisualShape(self.block, -1,
                                       rgbaColor=np.append(variable_value,
                                                           self.alpha))
        #TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
            state["silhouette_" + self.name + "_type"] = self.type_id
            state["silhouette_" + self.name + "_position"] = np.array(position)
            state["silhouette_" + self.name + "_orientation"] = np.array(orientation)
            state["silhouette_" + self.name + "_size"] = self.size
            state["silhouette_" + self.name + "_colour"] = self.colour
        elif state_type == 'list':
            state = []
            position, orientation = pybullet.getBasePositionAndOrientation(
                self.block
            )
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self.type_id)
                elif name == 'position':
                    state.extend(position)
                elif name == 'orientation':
                    state.extend(orientation)
                elif name == 'size':
                    state.extend(self.size)
                elif name == 'colour':
                    state.extend(self.colour)
        else:
            raise Exception("state type is not supported")
        return state

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_state_variable_names(self):
        return self._state_variable_names

    def get_state_size(self):
        return self.state_size

    def set_pose(self, position, orientation):
        pybullet.resetBasePositionAndOrientation(
            self.block, position, orientation
        )
        return
