import pybullet
import numpy as np


class SilhouetteObject(object):
    def __init__(self, pybullet_client, name, block_id):
        self.pybullet_client = pybullet_client
        self.name = name
        self.block_id = block_id

    def get_state(self, state_type='dict'):
        raise NotImplementedError()

    def set_state(self, state_dict):
        raise NotImplementedError()

    def set_full_state(self, new_state):
        raise NotImplementedError()

    def get_bounds(self):
        raise NotImplementedError()

    def do_intervention(self, variable_name, variable_value):
        raise NotImplementedError()

    def get_state_variable_names(self):
        raise NotImplementedError()

    def get_state_size(self):
        raise NotImplementedError()

    def set_pose(self, position, orientation):
        raise NotImplementedError()

    def get_variable_state(self, variable_name):
        raise NotImplementedError()

    def get_bounding_box(self):
        #TODO: this returns a point instead
        return self.pybullet_client.getAABB(self.block_id)


class SCuboid(SilhouetteObject):
    def __init__(
            self,
            pybullet_client,
            name, size=np.array([0.065, 0.065, 0.065]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            alpha=0.3, colour=np.array([0, 1, 0])
    ):
        self.type_id = 20
        self.size = size
        self.colour = colour
        self.alpha = alpha
        self.shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size) / 2,
            rgbaColor=np.append(self.colour, alpha)
        )
        self.block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        super(SCuboid, self).__init__(pybullet_client, name, self.block_id)
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
            np.array([-0.5, -0.5, 0])
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
        self.position = position
        self.orientation = orientation

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
        if 'position' in state_dict:
            self.position = state_dict['position']
        if 'orientation' in state_dict:
            self.orientation = state_dict['orientation']
        if 'size' in state_dict:
            self.size = np.array(state_dict['size'])
            self.pybullet_client.removeBody(self.block_id)
            self.shape_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=self.size / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation
            )
        elif 'position' in state_dict or 'orientation' in state_dict:
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, self.orientation
            )
        if 'colour' in state_dict:
            self.colour = state_dict['colour']
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(state_dict['colour'], self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, variable_value, self.orientation
            )
            self.position = variable_value
        elif variable_name == 'orientation':
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, variable_value
            )
            self.orientation = variable_value
        elif variable_name == 'size':
            self.pybullet_client.removeBody(self.block_id)
            self.shape_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(variable_value) / 2,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation
            )
            self.size = variable_value
        elif variable_name == 'colour':
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(variable_value,
                                                   self.alpha))
            self.colour = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            state[self.name + "_type"] = self.type_id
            state[self.name + "_position"] = np.array(self.position)
            state[self.name + "_orientation"] = np.array(self.orientation)
            state[self.name + "_size"] = self.size
            state[self.name + "_colour"] = self.colour
        elif state_type == 'list':
            state = []
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self.type_id)
                elif name == 'position':
                    state.extend(self.position)
                elif name == 'orientation':
                    state.extend(self.orientation)
                elif name == 'size':
                    state.extend(self.size)
                elif name == 'colour':
                    state.extend(self.colour)
        else:
            raise Exception("state type is not supported")
        return state

    def get_variable_state(self, variable_name):
        if variable_name == 'type':
            return self.type_id
        elif variable_name == 'position':
            return self.position
        elif variable_name == 'orientation':
            return self.orientation
        elif variable_name == 'size':
            return self.size
        elif variable_name == 'colour':
            return self.colour
        else:
            raise Exception("variable name is not supported")

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_state_variable_names(self):
        return self._state_variable_names

    def get_state_size(self):
        return self.state_size

    def set_pose(self, position, orientation):
        self.pybullet_client.resetBasePositionAndOrientation(
            self.block_id, position, orientation
        )
        self.position = position
        self.orientation = orientation
        return


class SSphere(SilhouetteObject):
    def __init__(
            self,
            pybullet_client,
            name, radius=np.array([0.015]),
            position=np.array([0.0, 0.0, 0.0425]),
            alpha=0.3, colour=np.array([0, 1, 0])
    ):
        self.type_id = 21
        self.radius = radius
        self.colour = colour
        self.alpha = alpha
        self.shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=radius,
            rgbaColor=np.append(self.colour, alpha)
        )
        self.block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        super(SSphere, self).__init__(pybullet_client, name, self.block_id)
        self.state = dict()
        self.state[self.name + "_type"] = self.type_id
        self.state[self.name + "_position"] = position
        self.state[self.name + "_radius"] = self.radius
        self.state[self.name + "_colour"] = self.colour

        # specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self.name + "_type"] = np.array([0])
        self.lower_bounds[self.name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self.name + "_radius"] = \
            np.array([0.01, 0.01, 0.01])
        self.lower_bounds[self.name + "_colour"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds[self.name + "_type"] = np.array([10])
        self.upper_bounds[self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_radius"] = \
            np.array([0.3, 0.3, 0.3])
        self.upper_bounds[self.name + "_colour"] = \
            np.array([1] * 3)

        self._state_variable_names = ['type', 'position',
                                      'radius', 'colour']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self.name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]
        self.position = position

    def set_full_state(self, new_state):
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
        if 'position' in state_dict:
            self.position = state_dict['position']
        if 'radius' in state_dict:
            self.radius = np.array(state_dict['radius'])
            self.pybullet_client.removeBody(self.block_id)
            self.shape_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=self.radius,
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=[0, 0, 0, 1]
            )
        elif 'position' in state_dict or 'orientation' in state_dict:
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, [0, 0, 0, 1]
            )
        if 'colour' in state_dict:
            self.colour = state_dict['colour']
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(state_dict['colour'], self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, variable_value, [0, 0, 0, 1]
            )
            self.position = variable_value
        elif variable_name == 'radius':
            self.pybullet_client.removeBody(self.block_id)
            self.shape_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=np.array(variable_value),
                rgbaColor=np.append(self.colour, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=[0, 0, 0, 1]
            )
            self.radius = np.array(variable_value)
        elif variable_name == 'colour':
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(variable_value,
                                                   self.alpha))
            self.colour = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            state[self.name + "_type"] = self.type_id
            state[self.name + "_position"] = np.array(self.position)
            state[self.name + "_radius"] = self.radius
            state[self.name + "_colour"] = self.colour
        elif state_type == 'list':
            state = []
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self.type_id)
                elif name == 'position':
                    state.extend(self.position)
                elif name == 'radius':
                    state.extend(self.radius)
                elif name == 'colour':
                    state.extend(self.colour)
        else:
            raise Exception("state type is not supported")
        return state

    def get_variable_state(self, variable_name):
        if variable_name == 'type':
            return self.type_id
        elif variable_name == 'position':
            return self.position
        elif variable_name == 'radius':
            return self.radius
        elif variable_name == 'colour':
            return self.colour
        else:
            raise Exception("variable name is not supported")

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_state_variable_names(self):
        return self._state_variable_names

    def get_state_size(self):
        return self.state_size

    def set_pose(self, position, orientation):
        self.pybullet_client.resetBasePositionAndOrientation(
            self.block_id, position, [0, 0, 0, 1]
        )
        self.position = position
        return
