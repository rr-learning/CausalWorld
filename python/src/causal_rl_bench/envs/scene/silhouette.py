import pybullet
import numpy as np
from causal_rl_bench.utils.rotation_utils import rotate_points


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

    def get_area(self):
        raise NotImplementedError()


class SCuboid(SilhouetteObject):
    def __init__(
            self,
            pybullet_client,
            name, size=np.array([0.065, 0.065, 0.065]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            alpha=0.3, color=np.array([0, 1, 0])
    ):
        self.type_id = 20
        self.size = size
        self.color = color
        self.alpha = alpha
        self.shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size) / 2,
            rgbaColor=np.append(self.color, alpha)
        )
        self.block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        super(SCuboid, self).__init__(pybullet_client, name, self.block_id)

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
        self.lower_bounds[self.name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds[self.name + "_type"] = np.array([10])
        self.upper_bounds[self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_orientation"] = \
            np.array([10] * 4)
        self.upper_bounds[self.name + "_size"] = \
            np.array([0.3, 0.3, 0.3])
        self.upper_bounds[self.name + "_color"] = \
            np.array([1] * 3)

        self._state_variable_names = ['type', 'position',
                                      'orientation', 'size', 'color']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self.name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]
        self.position = position
        self.orientation = orientation
        self.vertices = None
        self.bounding_box = None
        self._set_vertices()
        self._set_bounding_box()
        self.area = None
        self._set_area()

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
                rgbaColor=np.append(self.color, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation
            )
            self._set_vertices()
            self._set_bounding_box()
            self._set_area()
        elif 'position' in state_dict or 'orientation' in state_dict:
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, self.orientation
            )
            self._set_vertices()
            self._set_bounding_box()
        if 'color' in state_dict:
            self.color = state_dict['color']
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=
                                                   np.append(
                                                       state_dict['color'],
                                                       self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, variable_value, self.orientation
            )
            self.position = variable_value
            self._set_vertices()
            self._set_bounding_box()
        elif variable_name == 'orientation':
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, variable_value
            )
            self.orientation = variable_value
            self._set_vertices()
            self._set_bounding_box()
        elif variable_name == 'size':
            self.pybullet_client.removeBody(self.block_id)
            self.shape_id = self.pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(variable_value) / 2,
                rgbaColor=np.append(self.color, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation
            )
            self.size = variable_value
            self._set_vertices()
            self._set_bounding_box()
            self._set_area()
        elif variable_name == 'color':
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(variable_value,
                                                   self.alpha))
            self.color = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            state["type"] = self.type_id
            state["position"] = np.array(self.position)
            state["orientation"] = np.array(self.orientation)
            state["size"] = self.size
            state["color"] = self.color
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
                elif name == 'color':
                    state.extend(self.color)
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
        elif variable_name == 'color':
            return self.color
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
        self._set_vertices()
        self._set_bounding_box()
        return

    def _set_vertices(self):
        vertices = [[1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1]]
        vertices = [self.position + (point*self.size/2)
                    for point in vertices]
        self.vertices = rotate_points(np.array(vertices), self.orientation)
        return

    def _set_bounding_box(self):
        # low values for each axis
        low_bound = [np.min(self.vertices[:, 0]), np.min(self.vertices[:, 1]),
                     np.min(self.vertices[:, 2])]
        upper_bound = [np.max(self.vertices[:, 0]),
                       np.max(self.vertices[:, 1]),
                       np.max(self.vertices[:, 2])]
        self.bounding_box = (tuple(low_bound), tuple(upper_bound))

    def get_bounding_box(self):
        #low values for each axis
        return self.bounding_box

    def _set_area(self):
        self.area = self.size[0] * self.size[1] * self.size[2]
        return

    def get_area(self):
        return self.area


class SSphere(SilhouetteObject):
    #TODO: implement get bounding box and get area
    def __init__(
            self,
            pybullet_client,
            name, radius=np.array([0.015]),
            position=np.array([0.0, 0.0, 0.0425]),
            alpha=0.3, color=np.array([0, 1, 0])
    ):
        self.type_id = 21
        self.radius = radius
        self.color = color
        self.alpha = alpha
        self.shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=radius,
            rgbaColor=np.append(self.color, alpha)
        )
        self.block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        super(SSphere, self).__init__(pybullet_client, name, self.block_id)

        # specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self.name + "_type"] = np.array([0])
        self.lower_bounds[self.name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self.name + "_radius"] = \
            np.array([0.01])
        self.lower_bounds[self.name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds[self.name + "_type"] = np.array([10])
        self.upper_bounds[self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_radius"] = \
            np.array([0.3])
        self.upper_bounds[self.name + "_color"] = \
            np.array([1] * 3)

        self._state_variable_names = ['type', 'position',
                                      'radius', 'color']
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
            if not np.all(current_state[self._state_variable_names[i]] ==
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
                rgbaColor=np.append(self.color, self.alpha)
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
        if 'color' in state_dict:
            self.color = state_dict['color']
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(state_dict['color'], self.alpha))
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
                rgbaColor=np.append(self.color, self.alpha)
            )
            self.block_id = self.pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=[0, 0, 0, 1]
            )
            self.radius = np.array(variable_value)
        elif variable_name == 'color':
            self.pybullet_client.changeVisualShape(self.block_id, -1,
                                                   rgbaColor=np.append(variable_value,
                                                   self.alpha))
            self.color = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """
        Returns:
            Current position and orientation of the block.
        """
        if state_type == 'dict':
            state = dict()
            state["type"] = self.type_id
            state["position"] = np.array(self.position)
            state["radius"] = self.radius
            state["color"] = self.color
        elif state_type == 'list':
            state = []
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self.type_id)
                elif name == 'position':
                    state.extend(self.position)
                elif name == 'radius':
                    state.extend(self.radius)
                elif name == 'color':
                    state.extend(self.color)
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
        elif variable_name == 'color':
            return self.color
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
