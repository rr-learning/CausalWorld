import pybullet
import numpy as np
import copy
from causal_rl_bench.utils.rotation_utils import rotate_points


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
        position, orientation = \
            pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
        vertices = [[1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1]]
        vertices = [position + (point * self._size / 2)
                    for point in vertices]
        return rotate_points(np.array(vertices), orientation)

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
        self._type_id = 20
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


class SSphere(SilhouetteObject):
    def __init__(
            self,
            pybullet_client_ids,
            name, radius=0.015,
            position=np.array([0.0, 0.0, 0.0425]),
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
        self._radius = radius
        size = np.array([self._radius * 2, self._radius * 2, self._radius * 2])
        super(SSphere, self).__init__(pybullet_client_ids=pybullet_client_ids,
                                      name=name,
                                      size=size,
                                      position=position,
                                      orientation=[0, 0, 0, 1],
                                      color=color)

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=self._radius,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id
        )
        block_id = pybullet.createMultiBody(
            baseVisualShapeIndex=shape_id,
            basePosition=self._position,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=pybullet_client_id
        )
        return shape_id, block_id

    def _define_type_id(self):
        self._type_id = 21
        return

    def apply_interventions(self, interventions_dict):
        if 'size' in interventions_dict:
            raise Exception("can't apply intervention on size")
        if 'orientation' in interventions_dict:
            raise Exception("can't apply intervention on orientation")
        super(SSphere, self).apply_interventions(interventions_dict)
        #TODO: handle intervention on radius
        return

    def get_recreation_params(self):
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['radius'] = self._radius
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        recreation_params['position'] = position
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)


class SMeshObject(SilhouetteObject):
    def __init__(
        self,
        pybullet_client_ids, name,
        filename,
        scale=np.array([0.01, 0.01, 0.01]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        color=np.array([0, 1, 0])
    ):
        """

        :param pybullet_client:
        :param name:
        :param size:
        :param position:
        :param orientation:
        :param mass:
        :param color:
        """
        #TODO: intervene on friction as well
        self._scale = scale
        self._filename = filename
        size = self._set_size(pybullet_client_ids, position, orientation)
        super(SMeshObject, self).__init__(pybullet_client_ids=
                                          pybullet_client_ids,
                                          name=name,
                                          size=size,
                                          position=position,
                                          orientation=orientation,
                                          color=color)

    def _set_size(self, pybullet_client_ids, position, orientation):
        """

        :param pybullet_client:
        :return:
        """
        temp_shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=self._scale,
            fileName=self._filename,
            physicsClientId=pybullet_client_ids[0])
        temp_block_id = pybullet.createMultiBody(
            baseCollisionShapeIndex=temp_shape_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=0.1,
            physicsClientId=pybullet_client_ids[0])
        bb = pybullet.getAABB(temp_block_id,
                              physicsClientId=pybullet_client_ids[0])
        size = np.array([(bb[1][0] -
                          bb[0][0]),
                        (bb[1][1] -
                         bb[0][1]),
                        (bb[1][2] -
                         bb[0][2])])
        pybullet.removeBody(temp_block_id,
                            physicsClientId=pybullet_client_ids[0])
        return size

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=self._scale,
            fileName=self._filename,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(
            baseVisualShapeIndex=shape_id,
            basePosition=self._position,
            baseOrientation=self._orientation,
            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        self._type_id = 23
        return

    def get_recreation_params(self):
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['filename'] = self._filename
        recreation_params['scale'] = self._scale
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        recreation_params['position'] = position
        recreation_params['orientation'] = orientation
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)
