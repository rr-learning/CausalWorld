import pybullet
import numpy as np
import copy
from causal_world.utils.rotation_utils import rotate_points, cyl2cart, \
    cart2cyl, euler_to_quaternion
from causal_world.configs.world_constants import WorldConstants


class SilhouetteObject(object):

    def __init__(self, pybullet_client_ids, name, size, position, orientation,
                 color):
        """
        This is the base object for a silhouette in the arena.

        :param pybullet_client_ids: (list) list of pybullet client ids.
        :param name: (str) specifies the name of the silhouette object
        :param size: (list float) specifies the size of the object.
        :param position: (list float) x, y, z position.
        :param orientation: (list float) quaternion.
        :param color: (list float) RGB values.
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
        self._lower_bounds = dict()
        self._upper_bounds = dict()
        self._lower_bounds[self._name + "_type"] = \
            np.array([self._type_id])
        self._lower_bounds[self._name + "_cartesian_position"] = \
            np.array([-0.5, -0.5, 0])
        self._lower_bounds[self._name + "_cylindrical_position"] = \
            np.array([0, 0, 0])
        self._lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self._lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self._lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        #decision: type id is not normalized
        self._upper_bounds[self._name + "_type"] = \
            np.array([self._type_id])
        self._upper_bounds[self._name + "_cartesian_position"] = \
            np.array([0.5] * 3)
        self._upper_bounds[self._name + "_cylindrical_position"] = \
            np.array([0.20, np.pi, 0.5])
        self._upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self._upper_bounds[self._name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self._upper_bounds[self._name + "_color"] = \
            np.array([1] * 3)
        self._state_variable_names = []
        self._state_variable_names = [
            'type', 'cartesian_position', 'cylindrical_position', 'orientation',
            'size', 'color'
        ]
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
            sets the volume of the goal using the size attribute.

        :return:
        """
        self._volume = self._size[0] * self._size[1] * self._size[2]
        return

    def _add_state_variables(self):
        """
            used to add state variables to the silhouette object.

        :return:
        """
        return

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

        :param pybullet_client_id: (int) pybullet client id to be used when
                                         creating the gaol itself.
        :param kwargs: (params) parameters to be used when creating the goal
                                using the corresponding pybullet goal creation
                                parameters

        :return:
        """
        raise NotImplementedError("the creation function is not defined "
                                  "yet")

    def _define_type_id(self):
        """
            Defines the type id of the goal itself.

        :return:
        """
        raise NotImplementedError("the define type id function "
                                  "is not defined yet")

    def _init_object(self):
        """
            Used to initialize the goal, by creating it in the arena.
        :return:
        """
        for pybullet_client_id in self._pybullet_client_ids:
            shape_id, block_id =\
                self._create_object(pybullet_client_id)
            self._block_ids.append(block_id)
            self._shape_ids.append(shape_id)
        self._set_color(self._color)
        return

    def reinit_object(self):
        """
            Used to remove the goal from the arena and creating it again.

        :return:
        """
        self.remove()
        self._init_object()
        return

    def remove(self):
        """
            Used to remove the goal from the arena.

        :return:
        """
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.removeBody(self._block_ids[i],
                                physicsClientId=self._pybullet_client_ids[i])
        self._block_ids = []
        self._shape_ids = []
        return

    def _set_color(self, color):
        """

        :param color: (list) color RGB normalized from 0 to 1.

        :return:
        """
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeVisualShape(
                self._block_ids[i],
                -1,
                rgbaColor=np.append(color, self._alpha),
                physicsClientId=self._pybullet_client_ids[i])
        return

    def set_pose(self, position, orientation):
        """

        :param position: (list) cartesian x,y,z positon of the center of the
                                gaol shape.
        :param orientation: (list) quaternion x,y,z,w of the goal itself.

        :return:
        """
        position[-1] += WorldConstants.FLOOR_HEIGHT
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.resetBasePositionAndOrientation(
                self._block_ids[i],
                position,
                orientation,
                physicsClientId=self._pybullet_client_ids[i])
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type: (str) specifying 'dict' or 'list'

        :return: (list) returns either a dict or a list specifying the state
                        variables of the goal shape.
        """
        if state_type == 'dict':
            state = dict()
            position, orientation = \
                pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId =
                self._pybullet_client_ids[0])
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            state["type"] = self._type_id
            state["cartesian_position"] = np.array(position)
            state["cylindrical_position"] = cart2cyl(np.array(position))
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
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self._type_id)
                elif name == 'cartesian_position':
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

        :param variable_name: (str) specifies the variable name of the goal.

        :return: (nd.array) the high level variable value.
        """
        if variable_name == 'type':
            return self._type_id
        elif variable_name == 'cartesian_position':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0])
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            return position

        elif variable_name == 'cylindrical_position':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0])
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            return cart2cyl(position)

        elif variable_name == 'orientation':
            position, orientation = pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0])
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            return orientation
        elif variable_name == 'size':
            return self._size
        elif variable_name == 'color':
            return self._color

    def set_full_state(self, new_state):
        """

        :param new_state: (list) specifies the state to set the goal shape to.

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

        :param interventions_dict: (dict) specifies the interventions to be
                                          performed on the goal shape itself.

        :return:
        """
        #TODO: Add frictions to apply interventions
        if 'cylindrical_position' in interventions_dict:
            interventions_dict['cartesian_position'] = \
                cyl2cart(interventions_dict['cylindrical_position'])
        if 'euler_orientation' in interventions_dict:
            interventions_dict['orientation'] = euler_to_quaternion(
                interventions_dict['euler_orientation'])
        if 'cartesian_position' not in interventions_dict or \
                'orientation' not in interventions_dict:
            position, orientation = pybullet.\
                getBasePositionAndOrientation(self._block_ids[0],
                                              physicsClientId=
                                              self._pybullet_client_ids[0])
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
        if 'cartesian_position' in interventions_dict:
            position = interventions_dict['cartesian_position']
        if 'orientation' in interventions_dict:
            orientation = interventions_dict['orientation']
        if 'size' in interventions_dict:
            self._size = interventions_dict['size']
            self._set_volume()
            self.reinit_object()
        if 'cartesian_position' in interventions_dict or 'orientation' in \
                interventions_dict:
            for i in range(0, len(self._pybullet_client_ids)):
                position[-1] += WorldConstants.FLOOR_HEIGHT
                pybullet.resetBasePositionAndOrientation(
                    self._block_ids[i],
                    position,
                    orientation,
                    physicsClientId=self._pybullet_client_ids[i])
        if 'color' in interventions_dict:
            self._color = interventions_dict['color']
            self._set_color(self._color)
        return

    def get_state_variable_names(self):
        """

        :return: (list) specifies the various variable names in the list
                        itself.
        """
        return self._state_variable_names

    def get_bounds(self):
        """

        :return: (tuple) first position specifying the lower bounds of
                         the variables, second position specifying the upper
                         bounds of the variables.
        """
        return self._lower_bounds, self._upper_bounds

    def get_state_size(self):
        """

        :return: (int) returns the size of the state variables.
        """
        return self._state_size

    def get_bounding_box(self):
        """

        :return: (tuple) first position specifying the lower bounds of
                         the variables, second position specifying the upper
                         bounds of the variables.
        """
        vertices = self.get_vertices()
        # low values for each axis
        low_bound = [
            np.min(vertices[:, 0]),
            np.min(vertices[:, 1]),
            np.min(vertices[:, 2])
        ]
        upper_bound = [
            np.max(vertices[:, 0]),
            np.max(vertices[:, 1]),
            np.max(vertices[:, 2])
        ]
        return (tuple(low_bound), tuple(upper_bound))

    def get_vertices(self):
        """

        :return: (nd.array) specifies the current vertices of the gaol shape.
        """
        position, orientation = \
            pybullet.getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        vertices = [[1, 1, -1, 1], [1, -1, -1, 1], [-1, 1, -1, 1], [-1, -1, -1, 1],
                    [1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]]
        temp_size = np.array([self._size[0], self._size[1], self._size[2], 2])
        vertices = [(point * temp_size / 2.0) for point in vertices]
        return rotate_points(np.array(vertices), orientation, position)

    def get_size(self):
        """

        :return: (nd.array) returns the size of the goal shape.
        """
        return self._size

    def get_volume(self):
        """

        :return: (nd.array) returns the volume of the goal shape.
        """
        return self._volume

    def get_name(self):
        """

        :return: (str) returns the name of the object.
        """
        return self._name

    def get_block_ids(self):
        """

        :return: (list) returns the block ids in the active
                        pybullet clients.
        """
        return self._block_ids


class SCuboid(SilhouetteObject):

    def __init__(self,
                 pybullet_client_ids,
                 name,
                 size=np.array([0.065, 0.065, 0.065]),
                 position=np.array([0.0, 0.0, 0.0425]),
                 orientation=np.array([0, 0, 0, 1]),
                 color=np.array([0, 1, 0])):
        """
        This is the silhoutte cuboid object.

        :param pybullet_client_ids: (list) specifies the pybullet client ids.
        :param name: (str) specifies the name of the object.
        :param size: (list float) specifies the size in the three directions.
        :param position: (list float) specifies the position in x,y,z.
        :param orientation: (list float) specifies the quaternion of
                                         the object.
        :param color: (list float) specifies the RGB values of the cuboid.
        """
        super(SCuboid, self).__init__(pybullet_client_ids=pybullet_client_ids,
                                      name=name,
                                      size=size,
                                      position=position,
                                      orientation=orientation,
                                      color=color)

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

        :param pybullet_client_id: (list) specifies the pybullet client ids.
        :param kwargs: (params) parameters for the goal shape creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        position = np.array(self._position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=self._size / 2,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(baseVisualShapeIndex=shape_id,
                                            basePosition=position,
                                            baseOrientation=self._orientation,
                                            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the type id.

        :return:
        """
        self._type_id = 20
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the goal.
        """
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['size'] = self._size
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        recreation_params['position'] = position
        recreation_params['orientation'] = orientation
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)


class SSphere(SilhouetteObject):

    def __init__(self,
                 pybullet_client_ids,
                 name,
                 radius=0.015,
                 position=np.array([0.0, 0.0, 0.0425]),
                 color=np.array([0, 1, 0])):
        """

        :param pybullet_client_ids: (list) specifies the pybullet client ids.
        :param name: (str) specifies the name of the goal shape.
        :param radius: (float) specifies the radius of the sphere goal.
        :param position: (list float) specifies the position in x,y,z.
        :param color: (list float) specifies the RGB values of the cuboid.
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
        """

        :param pybullet_client_id: (int) corresponding pybullet client to create
                                         the gaol shape in.
        :param kwargs:  (params) parameters for the goal shape creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        position = np.array(self._position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=self._radius,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(baseVisualShapeIndex=shape_id,
                                            basePosition=position,
                                            baseOrientation=[0, 0, 0, 1],
                                            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the type id of the gaol shape.

        :return:
        """
        self._type_id = 21
        return

    def apply_interventions(self, interventions_dict):
        """

        :param interventions_dict: (dict) specifies the interventions to be
                                          performed on the various variables.

        :return:
        """
        if 'size' in interventions_dict:
            raise Exception("can't apply intervention on size")
        if 'orientation' in interventions_dict:
            raise Exception("can't apply intervention on orientation")
        super(SSphere, self).apply_interventions(interventions_dict)
        #TODO: handle intervention on radius
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the
                        goal shape.
        """
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['radius'] = self._radius
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        recreation_params['position'] = position
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)


class SMeshObject(SilhouetteObject):

    def __init__(self,
                 pybullet_client_ids,
                 name,
                 filename,
                 scale=np.array([0.01, 0.01, 0.01]),
                 position=np.array([0.0, 0.0, 0.0425]),
                 orientation=np.array([0, 0, 0, 1]),
                 color=np.array([0, 1, 0])):
        """

        :param pybullet_client_ids: (list) specifies the pybullet clients.
        :param name: (str) specifies the name of the goal.
        :param filename: (str) specifies the name of the file itself.
        :param scale: (list float) specifies the scale of the mesh goal.
        :param position: (list float) specifies the positions in x,y,z
        :param orientation: (list float) specifies the quaternion of the goal.
        :param color: (list float) specifies the RGB values.
        """
        #TODO: intervene on friction as well
        self._scale = scale
        self._filename = filename
        size = self._set_size(pybullet_client_ids, position, orientation)
        super(SMeshObject,
              self).__init__(pybullet_client_ids=pybullet_client_ids,
                             name=name,
                             size=size,
                             position=position,
                             orientation=orientation,
                             color=color)

    def _set_size(self, pybullet_client_ids, position, orientation):
        """

        :param pybullet_client_ids: (list) specifies the pybullet clients.
        :param position: (list float) specifies the positions in x,y,z
        :param orientation: (list float) specifies the quaternion of the goal.

        :return: (list) the size of the bounding box of the object.
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
        size = np.array([(bb[1][0] - bb[0][0]), (bb[1][1] - bb[0][1]),
                         (bb[1][2] - bb[0][2])])
        pybullet.removeBody(temp_block_id,
                            physicsClientId=pybullet_client_ids[0])
        return size

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

        :param pybullet_client_id: (list) specifies the pybullet clients.
        :param kwargs: (params) parameters for the goal shape creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        position = np.array(self._position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=self._scale,
            fileName=self._filename,
            rgbaColor=np.append(self._color, self._alpha),
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(baseVisualShapeIndex=shape_id,
                                            basePosition=position,
                                            baseOrientation=self._orientation,
                                            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the goal id of the object.

        :return:
        """
        self._type_id = 23
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the
                        goal shape.
        """
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['filename'] = self._filename
        recreation_params['scale'] = self._scale
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        recreation_params['position'] = position
        recreation_params['orientation'] = orientation
        recreation_params['color'] = self._color
        return copy.deepcopy(recreation_params)
