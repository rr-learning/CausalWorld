import pybullet
import numpy as np
from causal_rl_bench.utils.rotation_utils import rotate_points, \
    get_transformation_matrix, get_rotation_matrix
import copy


class SilhouetteObject(object):
    def __init__(self, pybullet_client_ids, name,
                 size, position,
                 orientation, color):
        """

        :param pybullet_clients:
        :param name:
        :param block_id:
        """
        self._pybullet_client_ids = pybullet_client_ids
        self._name = name
        self._type_id = None
        self._size = size
        self._color = color
        self._alpha = 0.3
        self._position = position
        self._orientation = orientation
        self._block_ids = []
        self._shape_ids = []
        self._define_type_id()
        self._volume = None
        self._set_volume()
        self._init_object()
        # specifying bounds
        self._lower_bounds = dict()
        self._upper_bounds = dict()
        self._lower_bounds[self._name + "_type"] = \
            np.array([self._type_id])
        self._lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self._lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self._lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self._lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        #decision: type id is not normalized
        self._upper_bounds[self._name + "_type"] = \
            np.array([self._type_id])
        self._upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self._upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self._upper_bounds[self._name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self._upper_bounds[self._name + "_color"] = \
            np.array([1] * 3)
        self._state_variable_names = []
        self._state_variable_names = ['type', 'position',
                                       'orientation',
                                       'size', 'color']
        self._state_variable_sizes = []
        self._state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self._upper_bounds[self._name + "_" +
                                   state_variable_name].shape[0])
            self._state_size += self._state_variable_sizes[-1]
        self._add_state_variables()
        return

    def _set_volume(self):
        """

        :return:
        """
        self._volume = self._size[0] * self._size[1] * self._size[2]
        return

    def _add_state_variables(self):
        return

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        raise NotImplementedError("the creation function is not defined "
                                  "yet")

    def _define_type_id(self):
        raise NotImplementedError("the define type id function "
                                  "is not defined yet")

    def _init_object(self):
        for pybullet_client_id in self._pybullet_client_ids:
            shape_id, block_id =\
                self._create_object(pybullet_client_id)
            self._block_ids.append(block_id)
            self._shape_ids.append(shape_id)
        self._set_color(self._color)
        return

    def reinit_object(self):
        self.remove()
        self._init_object()
        return

    def remove(self):
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.removeBody(self._block_ids[i],
                                 physicsClientId=
                                 self._pybullet_client_ids[i]
                                 )
        self._block_ids = []
        self._shape_ids = []
        return

    def _set_color(self, color):
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeVisualShape(self._block_ids[i],
                                        -1,
                                        rgbaColor=np.append(
                                        color, self._alpha),
                                       physicsClientId=
                                       self._pybullet_client_ids[i]
                                       )
        return

    def set_pose(self, position, orientation):
        """

        :param position:
        :param orientation:
        :return:
        """
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.resetBasePositionAndOrientation(
                self._block_ids[i], position, orientation,
                physicsClientId=self._pybullet_client_ids[i]
            )
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
        if state_type == 'dict':
            state = dict()
            position, orientation = \
                pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId =
                self._pybullet_client_ids[0])
            state["type"] = self._type_id
            state["position"] = np.array(position)
            state["orientation"] = np.array(orientation)
            state["size"] = self._size
            state["color"] = self._color
        elif state_type == 'list':
            state = []

            position, orientation = pybullet.\
                getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self._type_id)
                elif name == 'position':
                    state.extend(position)
                elif name == 'orientation':
                    state.extend(orientation)
                elif name == 'size':
                    state.extend(self._size)
                elif name == 'color':
                    state.extend(self._color)
        return state

    def get_variable_state(self, variable_name):
        """

        :param variable_name:
        :return:
        """
        if variable_name == 'type':
            return self._type_id
        elif variable_name == 'position':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self._block_ids[0], physicsClientId=self._pybullet_client_ids[0]
            )
            return position

        elif variable_name == 'orientation':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0]
            )
            return orientation
        elif variable_name == 'size':
            return self._size
        elif variable_name == 'color':
            return self._color

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        #form dict first
        new_state_dict = dict()
        current_state = self.get_state()
        start = 0
        for i in range(len(self._state_variable_sizes)):
            end = start + self._state_variable_sizes[i]
            if not np.all(current_state[self._state_variable_names[i]] ==
                          new_state[start:end]):
                if end == start + 1:
                    new_state_dict[self._state_variable_names[i]] = \
                        new_state[start:end][0]
                else:
                    new_state_dict[self._state_variable_names[i]] = \
                        new_state[start:end]
            start = end
        self.apply_interventions(new_state_dict)
        return

    def apply_interventions(self, interventions_dict):
        """

        :param state_dict:
        :return:
        """
        #TODO: Add frictions to apply interventions
        if 'position' not in interventions_dict or \
                'orientation' not in interventions_dict:
            position, orientation = pybullet.\
                getBasePositionAndOrientation(self._block_ids[0],
                                              physicsClientId=
                                              self._pybullet_client_ids[0])
        if 'position' in interventions_dict:
            position = interventions_dict['position']
        if 'orientation' in interventions_dict:
            orientation = interventions_dict['orientation']
        if 'size' in interventions_dict:
            self._size = interventions_dict['size']
            self._set_volume()
            self.reinit_object()
        elif 'position' in interventions_dict or 'orientation' in \
                interventions_dict:
            for i in range(0, len(self._pybullet_client_ids)):
                pybullet.resetBasePositionAndOrientation(
                    self._block_ids[i], position, orientation,
                    physicsClientId=
                    self._pybullet_client_ids[i])
        if 'color' in interventions_dict:
            self._color = interventions_dict['color']
            self._set_color(self._color)
        return

    def get_state_variable_names(self):
        """

        :return:
        """
        return self._state_variable_names

    def get_bounds(self):
        """

        :return:
        """
        return self._lower_bounds, self._upper_bounds

    def get_state_size(self):
        """

        :return:
        """
        return self._state_size

    def get_bounding_box(self):
        """

        :return:
        """
        vertices = self.get_vertices()
        # low values for each axis
        low_bound = [np.min(vertices[:, 0]),
                     np.min(vertices[:, 1]),
                     np.min(vertices[:, 2])]
        upper_bound = [np.max(vertices[:, 0]),
                       np.max(vertices[:, 1]),
                       np.max(vertices[:, 2])]
        return (tuple(low_bound), tuple(upper_bound))

    def get_vertices(self):
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
        vertices = [self._position + (point * self._size / 2)
                    for point in vertices]
        return rotate_points(np.array(vertices), self._orientation)

    def get_size(self):
        """

        :return:
        """
        return self._size

    def get_volume(self):
        """

        :return:
        """
        return self._volume

    def get_name(self):
        """

        :return:
        """
        return self._name

    def get_block_ids(self):
        """

        :return:
        """
        return self._block_ids


class SCuboid(SilhouetteObject):
    def __init__(
            self,
            pybullet_client_ids,
            name, size=np.array([0.065, 0.065, 0.065]),
            position=np.array([0.0, 0.0, 0.0425]),
            orientation=np.array([0, 0, 0, 1]),
            color=np.array([0, 1, 0])
    ):
        """

        :param pybullet_clients:
        :param name:
        :param size:
        :param position:
        :param orientation:
        :param alpha:
        :param color:
        """
        super(SCuboid, self).__init__(pybullet_client_ids=pybullet_client_ids,
                                      name=name,
                                      size=size,
                                      position=position,
                                      orientation=orientation,
                                      color=color)

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=self._size / 2,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id
        )
        block_id = pybullet.createMultiBody(
            baseVisualShapeIndex=shape_id,
            basePosition=self._position,
            baseOrientation=self._orientation,
            physicsClientId=pybullet_client_id
        )
        return shape_id, block_id

    def _define_type_id(self):
        self._type_id = 60
        return

    def get_recreation_params(self):
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['size'] = self._size
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        recreation_params['position'] = position
        recreation_params['orientation'] = orientation
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)


class SMeshObject(SilhouetteObject):
    def __init__(
            self,
            pybullet_clients,
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
        self.__set_size(pybullet_clients)
        self.__block_ids = []
        self.__shape_ids = []
        for pybullet_client in pybullet_clients:
            __shape_id = pybullet_client.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=scale,
            fileName=filename,
            rgbaColor=np.append(self.__color, alpha)
        )
            __block_id = pybullet_client.createMultiBody(
                baseVisualShapeIndex=__shape_id,
                basePosition=position,
                baseOrientation=orientation,
            )
            self.__block_ids.append(__block_id)
            self.__shape_ids.append(__shape_id)
        super(SMeshObject, self).__init__(pybullet_clients, name,
                                          self.__block_ids)

        # specifying bounds
        self.__lower_bounds = dict()
        self.__upper_bounds = dict()
        self.__lower_bounds[self._name + "_type"] = np.array([0])
        self.__lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.__lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.__lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self.__lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        # TODO: the type id here is arbitrary, need to be changed
        self.__upper_bounds[self._name + "_type"] = np.array([10])
        self.__upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self.__upper_bounds[self._name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
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
            self.__position = np.array(state_dict['position'])
        if 'orientation' in state_dict:
            self.__orientation = np.array(state_dict['orientation'])
        elif 'position' in state_dict or 'orientation' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.__position, self.__orientation
                )
            self.__set_vertices()
            self.__set_bounding_box()
        if 'color' in state_dict:
            self.__color = np.array(state_dict['color'])
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(self.__block_ids[i], -1,
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
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], variable_value, self.__orientation
                )
            self.__position = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'orientation':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.__position, variable_value
                )
            self.__orientation = variable_value
            self.__set_vertices()
            self.__set_bounding_box()
        elif variable_name == 'color':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(self.__block_ids[i],
                                                            -1,
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
        for i in range(0, len(self._pybullet_clients)):
            self._pybullet_clients[i].resetBasePositionAndOrientation(
                self.__block_ids[i], position, orientation
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

    def __set_size(self, pybullet_clients):
        """

        :param pybullet_client:
        :return:
        """
        temp_shape_id = pybullet_clients[0].createCollisionShape(
            shapeType=pybullet_clients.GEOM_MESH,
            meshScale=self.__scale,
            fileName=self.__filename)
        temp_block_id = pybullet_clients[0].createMultiBody(
            baseCollisionShapeIndex=temp_shape_id,
            basePosition=self.__position,
            baseOrientation=self.__orientation,
            baseMass=0.1
        )
        self.__bounding_box = pybullet_clients[0].getAABB(temp_block_id)
        self.__size = np.array([(self.__bounding_box[1][0] -
                                 self.__bounding_box[0][0]),
                                (self.__bounding_box[1][1] -
                                 self.__bounding_box[0][1]),
                                (self.__bounding_box[1][2] -
                                 self.__bounding_box[0][2])])
        pybullet_clients[0].removeBody(temp_block_id)
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
            pybullet_clients,
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

        self.__block_ids = []
        self.__shape_ids = []
        for pybullet_client in pybullet_clients:
            __shape_id = pybullet_client.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=radius,
                rgbaColor=np.append(self.color, alpha)
            )
            __block_id = pybullet_client.createMultiBody(
                baseVisualShapeIndex=__shape_id,
                basePosition=position,
                baseOrientation=[0, 0, 0, 1],
            )
            self.__block_ids.append(__block_id)
            self.__shape_ids.append(__shape_id)
        super(SSphere, self).__init__(pybullet_clients, name,
                                      self.__block_ids)

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
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].removeBody(self.__block_ids[i])
            self.__block_ids = []
            self.__shape_ids = []
            for i in range(0, len(self._pybullet_clients)):
                __shape_id = self._pybullet_clients[i].createVisualShape(
                    shapeType=pybullet.GEOM_SPHERE,
                    radius=self.radius,
                    rgbaColor=np.append(self.color, self.alpha)
                )
                __block_id = self._pybullet_clients[i].createMultiBody(
                    baseVisualShapeIndex=__shape_id,
                    basePosition=self.position,
                    baseOrientation=[0, 0, 0, 1]
                )
                self.__shape_ids.append(__shape_id)
                self.__block_ids.append(__block_id)
            self._set_bounding_box()
            self._set_volume()
        elif 'position' in state_dict or 'orientation' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.position, [0, 0, 0, 1]
                )
        if 'color' in state_dict:
            self.color = state_dict['color']
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].\
                    changeVisualShape(self.__block_ids[i], -1,
                                      rgbaColor=np.append(state_dict['color'],
                                                          self.alpha))
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        # TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], variable_value, [0, 0, 0, 1]
                )
                self.position = variable_value
        elif variable_name == 'radius':
            self.radius = np.array(variable_value)
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].removeBody(self.__block_ids[i])
            self.__block_ids = []
            self.__shape_ids = []
            for i in range(0, len(self._pybullet_clients)):
                __shape_id = self._pybullet_clients[i].createVisualShape(
                    shapeType=pybullet.GEOM_SPHERE,
                    radius=self.radius,
                    rgbaColor=np.append(self.color, self.alpha)
                )
                __block_id = self._pybullet_clients[i].createMultiBody(
                    baseVisualShapeIndex=__shape_id,
                    basePosition=self.position,
                    baseOrientation=[0, 0, 0, 1]
                )
                self.__shape_ids.append(__shape_id)
                self.__block_ids.append(__block_id)
            self._set_bounding_box()
            self._set_volume()
        elif variable_name == 'color':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(self.__block_ids[i], -1,
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
        for i in range(0, len(self._pybullet_clients)):
            self._pybullet_clients[i].resetBasePositionAndOrientation(
                self.__block_ids[i], position, [0, 0, 0, 1]
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
