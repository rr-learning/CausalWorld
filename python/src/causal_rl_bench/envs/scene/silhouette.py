import pybullet
import numpy as np


class SilhouetteObject(object):
    def __init__(self, pybullet_client, name):
        self.pybullet_client = pybullet_client
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
            pybullet_client,
            name, size=np.array([0.065, 0.065, 0.065]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            alpha=0.3, colour=np.array([0, 1, 0])
    ):
        super(SCuboid, self).__init__(pybullet_client, name)
        self.type_id = 0
        self.size = size
        self.colour = colour
        self.alpha = alpha
        self.block_id = self.pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size) / 2,
            rgbaColor=np.append(self.colour, alpha)
        )
        self.block = self.pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.block_id,
            basePosition=position,
            baseOrientation=orientation
        )
        self.state = dict()
        self.state[self.name + "_type"] = self.type_id
        self.state[self.name + "_position"] = position
        self.state[self.name + "_orientation"] = orientation
        self.state[self.name + "_size"] = self.size
        self.state[self.name + "_colour"] = self.colour

        # specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self.name + "_type"] = np.array([0])
        self.lower_bounds[self.name + "_position"] = \
            np.array([-0.5] * 3)
        self.lower_bounds[self.name + "_orientation"] = \
            np.array([-10] * 4)
        self.lower_bounds[self.name + "_size"] = \
            np.array([0.01, 0.01, 0.01])
        self.lower_bounds[self.name + "_colour"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds[self.name + "_type"] = np.array([10])
        self.upper_bounds[self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_orientation"] = \
            np.array([10] * 4)
        self.upper_bounds[self.name + "_size"] = \
            np.array([0.3, 0.3, 0.3])
        self.upper_bounds[self.name + "_colour"] = \
            np.array([1] * 3)

        self._state_variable_names = ['type', 'position',
                                      'orientation', 'size', 'colour']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self.name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]

    def set_full_state(self, new_state):
        # form dict first
        new_state_dict = dict()
        current_state = self.get_state()
        start = 0
        for i in range(len(self._state_variable_sizes)):
            end = start + self._state_variable_sizes[i]
            if not np.all(current_state[self.name + "_"
                                        + self._state_variable_names[i]] ==
                          new_state[start:end]):
                new_state_dict[self._state_variable_names[i]] = new_state[start:end]
            start = end
        self.set_state(new_state_dict)
        return

    def set_state(self, state_dict):
        if 'position' not in state_dict or 'orientation' not in state_dict:
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'size' in state_dict:
            self.pybullet_client.removeBody(self.block)
            self.block_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(state_dict['size']) / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation
            )
        elif 'position' in state_dict or 'orientation' in state_dict:
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block, position, orientation
            )
        if 'colour' in state_dict:
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor= np.append(state_dict['colour'], self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block, variable_value, orientation
            )
        elif variable_name == 'orientation':
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block, position, variable_value
            )
        elif variable_name == 'size':
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            self.pybullet_client.removeBody(self.block)
            self.block_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(variable_value) / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation
            )
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor=np.append(self.colour,
                                                   self.alpha))
        elif variable_name == 'colour':
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor=np.append(variable_value,
                                                   self.alpha))
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            state[self.name + "_type"] = self.type_id
            state[self.name + "_position"] = np.array(position)
            state[self.name + "_orientation"] = np.array(orientation)
            state[self.name + "_size"] = self.size
            state[self.name + "_colour"] = self.colour
        elif state_type == 'list':
            state = []
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
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
        self.pybullet_client.resetBasePositionAndOrientation(
            self.block, position, orientation
        )
        return
