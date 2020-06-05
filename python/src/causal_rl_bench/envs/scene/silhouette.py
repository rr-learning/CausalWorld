import pybullet
import numpy as np
from causal_rl_bench.utils.rotation_utils import rotate_points


class SilhouetteObject(object):
    def __init__(self, pybullet_client, name, block_id):
        """

        :param pybullet_client:
        :param name:
        :param block_id:
        """
        self._pybullet_client = pybullet_client
        self._name = name
        self._block_id = block_id

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
        raise NotImplementedError()

    def set_state(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        raise NotImplementedError()

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        raise NotImplementedError()

    def get_bounds(self):
        """

        :return:
        """
        raise NotImplementedError()

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        raise NotImplementedError()

    def get_state_variable_names(self):
        """

        :return:
        """
        raise NotImplementedError()

    def get_state_size(self):
        """

        :return:
        """
        raise NotImplementedError()

    def set_pose(self, position, orientation):
        """

        :param position:
        :param orientation:
        :return:
        """
        raise NotImplementedError()

    def get_variable_state(self, variable_name):
        """

        :param variable_name:
        :return:
        """
        raise NotImplementedError()

    def get_bounding_box(self):
        """

        :return:
        """
        raise NotImplementedError()

    def get_volume(self):
        """

        :return:
        """
        raise NotImplementedError()

    def get_name(self):
        """

        :return:
        """
        return self._name

    def get_block_id(self):
        """

        :return:
        """
        return self._block_id


class SCuboid(SilhouetteObject):
    def __init__(
            self,
            pybullet_client,
            name, size=np.array([0.065, 0.065, 0.065]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            alpha=0.3, color=np.array([0, 1, 0])
    ):
        """

        :param pybullet_client:
        :param name:
        :param size:
        :param position:
        :param orientation:
        :param alpha:
        :param color:
        """
        self.__type_id = 20
        self.__size = size
        self.__color = color
        self.__alpha = alpha
        self.__shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(size) / 2,
            rgbaColor=np.append(self.__color, alpha)
        )
        self.__block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.__shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        super(SCuboid, self).__init__(pybullet_client, name, self.__block_id)

        # specifying bounds
        self.__lower_bounds = dict()
        self.__upper_bounds = dict()
        self.__lower_bounds[self._name + "_type"] = np.array([0])
        self.__lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.__lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.__lower_bounds[self._name + "_size"] = \
            np.array([0.01, 0.01, 0.01])
        self.__lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.__upper_bounds[self._name + "_type"] = np.array([10])
        self.__upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self.__upper_bounds[self._name + "_size"] = \
            np.array([0.3, 0.3, 0.3])
        self.__upper_bounds[self._name + "_color"] = \
            np.array([1] * 3)

        self.__state_variable_names = ['type', 'position',
                                      'orientation', 'size', 'color']
        self.__state_variable_sizes = []
        self.__state_size = 0
        for state_variable_name in self.__state_variable_names:
            self.__state_variable_sizes.append(
                self.__upper_bounds[self._name + "_" +
                                    state_variable_name].shape[0])
            self.__state_size += self.__state_variable_sizes[-1]
        self.__position = position
        self.__orientation = orientation
        self.__vertices = None
        self.__bounding_box = None
        self.__set_vertices()
        self.__set_bounding_box()
        self.__volume = None
        self.__set_volume()

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        # form dict first
        new_state_dict = dict()
        current_state = self.get_state()
        start = 0
        for i in range(len(self.__state_variable_sizes)):
            end = start + self.__state_variable_sizes[i]
            if not np.all(current_state[self.__state_variable_names[i]] ==
                          new_state[start:end]):
                new_state_dict[self.__state_variable_names[i]] = \
                    new_state[start:end]
            start = end
        self.set_state(new_state_dict)
        return

    def set_state(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        if 'position' in state_dict:
            self.__position = state_dict['position']
        if 'orientation' in state_dict:
            self.__orientation = state_dict['orientation']
        if 'size' in state_dict:
            self.__size = np.array(state_dict['size'])
            self._pybullet_client.removeBody(self.__block_id)
            self.__shape_id = self._pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=self.__size / 2,
                rgbaColor=np.append(self.__color, self.__alpha)
            )
            self.__block_id = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.__shape_id,
                basePosition=self.__position,
                baseOrientation=self.__orientation
            )
            self.__set_vertices()
            self.__set_bounding_box()
            self.__set_volume()
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, self.__position, self.__orientation
            )
            self.__set_vertices()
            self.__set_bounding_box()
        if 'color' in state_dict:
            self.__color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=
                                                   np.append(
                                                       state_dict['color'],
                                                       self.__alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, variable_value, self.__orientation
            )
            self.__position = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'orientation':
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, self.__position, variable_value
            )
            self.__orientation = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'size':
            self._pybullet_client.removeBody(self.__block_id)
            self.__shape_id = self._pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(variable_value) / 2,
                rgbaColor=np.append(self.__color, self.__alpha)
            )
            self.__block_id = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.__shape_id,
                basePosition=self.__position,
                baseOrientation=self.__orientation
            )
            self.__size = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
            self.__set_volume()
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=np.append(variable_value,
                                                                        self.__alpha))
            self.__color = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
        if state_type == 'dict':
            state = dict()
            state["type"] = self.__type_id
            state["position"] = np.array(self.__position)
            state["orientation"] = np.array(self.__orientation)
            state["size"] = self.__size
            state["color"] = self.__color
        elif state_type == 'list':
            state = []
            for name in self.__state_variable_names:
                if name == 'type':
                    state.append(self.__type_id)
                elif name == 'position':
                    state.extend(self.__position)
                elif name == 'orientation':
                    state.extend(self.__orientation)
                elif name == 'size':
                    state.extend(self.__size)
                elif name == 'color':
                    state.extend(self.__color)
        else:
            raise Exception("state type is not supported")
        return state

    def get_variable_state(self, variable_name):
        """

        :param variable_name:
        :return:
        """
        if variable_name == 'type':
            return self.__type_id
        elif variable_name == 'position':
            return self.__position
        elif variable_name == 'orientation':
            return self.__orientation
        elif variable_name == 'size':
            return self.__size
        elif variable_name == 'color':
            return self.__color
        else:
            raise Exception("variable name is not supported")

    def get_bounds(self):
        """

        :return:
        """
        return self.__lower_bounds, self.__upper_bounds

    def get_state_variable_names(self):
        """

        :return:
        """
        return self.__state_variable_names

    def get_state_size(self):
        """

        :return:
        """
        return self.__state_size

    def set_pose(self, position, orientation):
        """

        :param position:
        :param orientation:
        :return:
        """
        self._pybullet_client.resetBasePositionAndOrientation(
            self.__block_id, position, orientation
        )
        self.__position = position
        self.__orientation = orientation
        self.__set_vertices()
        self.__set_bounding_box()
        return

    def __set_vertices(self):
        """

        :return:
        """
        vertices = [[1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1]]
        vertices = [self.__position + (point * self.__size / 2)
                    for point in vertices]
        self.__vertices = rotate_points(np.array(vertices), self.__orientation)
        return

    def __set_bounding_box(self):
        """

        :return:
        """
        # low values for each axis
        low_bound = [np.min(self.__vertices[:, 0]), np.min(self.__vertices[:, 1]),
                     np.min(self.__vertices[:, 2])]
        upper_bound = [np.max(self.__vertices[:, 0]),
                       np.max(self.__vertices[:, 1]),
                       np.max(self.__vertices[:, 2])]
        self.__bounding_box = (tuple(low_bound), tuple(upper_bound))

    def get_bounding_box(self):
        """

        :return:
        """
        #low values for each axis
        return self.__bounding_box

    def __set_volume(self):
        """

        :return:
        """
        self.__volume = self.__size[0] * self.__size[1] * self.__size[2]
        return

    def get_volume(self):
        """

        :return:
        """
        return self.__volume


class SMeshObject(SilhouetteObject):
    def __init__(
            self,
            pybullet_client,
            name, filename,
            scale=np.array([0.01, 0.01, 0.01]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            alpha=0.3, color=np.array([0, 1, 0])
    ):
        """

        :param pybullet_client:
        :param name:
        :param size:
        :param position:
        :param orientation:
        :param alpha:
        :param color:
        """
        self.__type_id = 20
        self.__scale = scale
        self.__color = color
        self.__alpha = alpha
        self.__position = position
        self.__orientation = orientation
        self.__filename = filename
        self.__size = None
        self.__set_size(pybullet_client)
        self.__shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=scale,
            fileName=filename,
            rgbaColor=np.append(self.__color, alpha)
        )
        self.__block_id = pybullet_client.createMultiBody(
            baseVisualShapeIndex=self.__shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        super(SMeshObject, self).__init__(pybullet_client, name,
                                          self.__block_id)

        # specifying bounds
        self.__lower_bounds = dict()
        self.__upper_bounds = dict()
        self.__lower_bounds[self._name + "_type"] = np.array([0])
        self.__lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.__lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.__lower_bounds[self._name + "_size"] = \
            np.array([0.01, 0.01, 0.01])
        self.__lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.__upper_bounds[self._name + "_type"] = np.array([10])
        self.__upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self.__upper_bounds[self._name + "_size"] = \
            np.array([0.3, 0.3, 0.3])
        self.__upper_bounds[self._name + "_color"] = \
            np.array([1] * 3)

        self.__state_variable_names = ['type', 'position',
                                       'orientation', 'size', 'color']
        self.__state_variable_sizes = []
        self.__state_size = 0
        for state_variable_name in self.__state_variable_names:
            self.__state_variable_sizes.append(
                self.__upper_bounds[self._name + "_" +
                                    state_variable_name].shape[0])
            self.__state_size += self.__state_variable_sizes[-1]
        self.__vertices = None
        self.__bounding_box = None
        self.__set_vertices()
        self.__set_bounding_box()
        self.__volume = None
        self.__set_volume()

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        # form dict first
        new_state_dict = dict()
        current_state = self.get_state()
        start = 0
        for i in range(len(self.__state_variable_sizes)):
            end = start + self.__state_variable_sizes[i]
            if not np.all(current_state[self.__state_variable_names[i]] ==
                          new_state[start:end]):
                new_state_dict[self.__state_variable_names[i]] = \
                    new_state[start:end]
            start = end
        self.set_state(new_state_dict)
        return

    def set_state(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        if 'position' in state_dict:
            self.__position = state_dict['position']
        if 'orientation' in state_dict:
            self.__orientation = state_dict['orientation']
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, self.__position, self.__orientation
            )
            self.__set_vertices()
            self.__set_bounding_box()
        if 'color' in state_dict:
            self.__color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=
                                                    np.append(
                                                        state_dict['color'],
                                                        self.__alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, variable_value, self.__orientation
            )
            self.__position = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'orientation':
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, self.__position, variable_value
            )
            self.__orientation = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=np.append(
                                                        variable_value,
                                                        self.__alpha))
            self.__color = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
        if state_type == 'dict':
            state = dict()
            state["type"] = self.__type_id
            state["position"] = np.array(self.__position)
            state["orientation"] = np.array(self.__orientation)
            state["size"] = self.__size
            state["color"] = self.__color
        elif state_type == 'list':
            state = []
            for name in self.__state_variable_names:
                if name == 'type':
                    state.append(self.__type_id)
                elif name == 'position':
                    state.extend(self.__position)
                elif name == 'orientation':
                    state.extend(self.__orientation)
                elif name == 'size':
                    state.extend(self.__size)
                elif name == 'color':
                    state.extend(self.__color)
        else:
            raise Exception("state type is not supported")
        return state

    def get_variable_state(self, variable_name):
        """

        :param variable_name:
        :return:
        """
        if variable_name == 'type':
            return self.__type_id
        elif variable_name == 'position':
            return self.__position
        elif variable_name == 'orientation':
            return self.__orientation
        elif variable_name == 'size':
            return self.__size
        elif variable_name == 'color':
            return self.__color
        else:
            raise Exception("variable name is not supported")

    def get_bounds(self):
        """

        :return:
        """
        return self.__lower_bounds, self.__upper_bounds

    def get_state_variable_names(self):
        """

        :return:
        """
        return self.__state_variable_names

    def get_state_size(self):
        """

        :return:
        """
        return self.__state_size

    def set_pose(self, position, orientation):
        """

        :param position:
        :param orientation:
        :return:
        """
        self._pybullet_client.resetBasePositionAndOrientation(
            self.__block_id, position, orientation
        )
        self.__position = position
        self.__orientation = orientation
        self.__set_vertices()
        self.__set_bounding_box()
        return

    def __set_vertices(self):
        """

        :return:
        """
        vertices = [[1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1]]
        vertices = [self.__position + (point * self.__size / 2)
                    for point in vertices]
        self.__vertices = rotate_points(np.array(vertices), self.__orientation)
        return

    def __set_size(self, pybullet_client):
        """

        :param pybullet_client:
        :return:
        """
        temp_shape_id = pybullet_client.createCollisionShape(
            shapeType=pybullet_client.GEOM_MESH,
            meshScale=self.__scale,
            fileName=self.__filename)
        temp_block_id = pybullet_client.createMultiBody(
            baseCollisionShapeIndex=temp_shape_id,
            basePosition=self.__position,
            baseOrientation=self.__orientation,
            baseMass=0.1
        )
        self.__bounding_box = pybullet_client.getAABB(temp_block_id)
        self.__size = np.array([(self.__bounding_box[1][0] -
                                 self.__bounding_box[0][0]),
                                (self.__bounding_box[1][1] -
                                 self.__bounding_box[0][1]),
                                (self.__bounding_box[1][2] -
                                 self.__bounding_box[0][2])])
        pybullet_client.removeBody(temp_block_id)
        return

    def __set_bounding_box(self):
        """

        :return:
        """
        # low values for each axis
        low_bound = [np.min(self.__vertices[:, 0]),
                     np.min(self.__vertices[:, 1]),
                     np.min(self.__vertices[:, 2])]
        upper_bound = [np.max(self.__vertices[:, 0]),
                       np.max(self.__vertices[:, 1]),
                       np.max(self.__vertices[:, 2])]
        self.__bounding_box = (tuple(low_bound), tuple(upper_bound))
        return

    def get_bounding_box(self):
        """

        :return:
        """
        # low values for each axis
        return self.__bounding_box

    def __set_volume(self):
        """

        :return:
        """
        self.__volume = self.__size[0] * self.__size[1] * self.__size[2]
        return

    def get_volume(self):
        """

        :return:
        """
        return self.__volume


class SSphere(SilhouetteObject):
    #TODO: implement get bounding box and get area
    def __init__(
            self,
            pybullet_client,
            name, radius=np.array([0.015]),
            position=np.array([0.0, 0.0, 0.0425]),
            alpha=0.3, color=np.array([0, 1, 0])
    ):
        """

        :param pybullet_client:
        :param name:
        :param radius:
        :param position:
        :param alpha:
        :param color:
        """
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
        self.lower_bounds[self._name + "_type"] = np.array([0])
        self.lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self._name + "_radius"] = \
            np.array([0.01])
        self.lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.upper_bounds[self._name + "_type"] = np.array([10])
        self.upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self._name + "_radius"] = \
            np.array([0.3])
        self.upper_bounds[self._name + "_color"] = \
            np.array([1] * 3)

        self._state_variable_names = ['type', 'position',
                                      'radius', 'color']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self._name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]
        self.position = position
        self.bounding_box = None
        self._set_bounding_box()
        self._volume = None
        self._set_volume()

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
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
        """

        :param state_dict:
        :return:
        """
        if 'position' in state_dict:
            self.position = state_dict['position']
            self._set_bounding_box()
        if 'radius' in state_dict:
            self.radius = np.array(state_dict['radius'])
            self._pybullet_client.removeBody(self.block_id)
            self.shape_id = self._pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=self.radius,
                rgbaColor=np.append(self.color, self.alpha)
            )
            self.block_id = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=[0, 0, 0, 1]
            )
            self._set_bounding_box()
            self._set_volume()
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, [0, 0, 0, 1]
            )
        if 'color' in state_dict:
            self.color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.block_id, -1,
                                                    rgbaColor=np.append(state_dict['color'], self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self._pybullet_client.resetBasePositionAndOrientation(
                self.block_id, variable_value, [0, 0, 0, 1]
            )
            self.position = variable_value
        elif variable_name == 'radius':
            self._pybullet_client.removeBody(self.block_id)
            self.shape_id = self._pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=np.array(variable_value),
                rgbaColor=np.append(self.color, self.alpha)
            )
            self.block_id = self._pybullet_client.createMultiBody(
                baseVisualShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=[0, 0, 0, 1]
            )
            self.radius = np.array(variable_value)
            self._set_bounding_box()
            self._set_volume()
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.block_id, -1,
                                                    rgbaColor=np.append(variable_value,
                                                   self.alpha))
            self.color = variable_value
        # TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
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
        """

        :param variable_name:
        :return:
        """
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
        """

        :return:
        """
        return self.lower_bounds, self.upper_bounds

    def get_state_variable_names(self):
        """

        :return:
        """
        return self._state_variable_names

    def get_state_size(self):
        """

        :return:
        """
        return self.state_size

    def set_pose(self, position, orientation):
        """

        :param position:
        :param orientation:
        :return:
        """
        self._pybullet_client.resetBasePositionAndOrientation(
            self.block_id, position, [0, 0, 0, 1]
        )
        self.position = position
        self._set_bounding_box()
        return

    def _set_bounding_box(self):
        """

        :return:
        """
        # low values for each axis
        low_bound = np.array(self.position) - \
                    np.array([self.radius, self.radius, self.radius]).flatten()
        upper_bound = np.array(self.position) + \
                     np.array([self.radius, self.radius, self.radius]).flatten()
        self.bounding_box = (tuple(low_bound), tuple(upper_bound))

    def get_bounding_box(self):
        """

        :return:
        """
        #low values for each axis
        return self.bounding_box

    def _set_volume(self):
        """

        :return:
        """
        self._volume = (self.radius**3) * 4/3.0 * np.pi
        return

    def get_volume(self):
        """

        :return:
        """
        return self._volume
