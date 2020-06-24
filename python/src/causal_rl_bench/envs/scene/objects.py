import pybullet
import numpy as np
from causal_rl_bench.utils.rotation_utils import rotate_points, \
    get_transformation_matrix, get_rotation_matrix
import copy


class RigidObject(object):
    def __init__(self, pybullet_client_ids, name,
                 size, initial_position,
                 initial_orientation, mass,
                 color, lateral_friction,
                 spinning_friction, restitution,
                 initial_linear_velocity,
                 initial_angular_velocity,
                 fixed_bool):
        """

        :param pybullet_clients:
        :param name:
        :param block_id:
        """
        self._pybullet_client_ids = pybullet_client_ids
        self._name = name
        self._type_id = None
        self._mass = mass
        self._size = size
        self._not_fixed = not fixed_bool
        self._color = color
        self._initial_position = initial_position
        self._initial_orientation = initial_orientation
        self._initial_linear_velocity = initial_linear_velocity
        self._initial_angular_velocity = initial_angular_velocity
        self._lateral_friction = lateral_friction
        self._spinning_friction = spinning_friction
        self._restitution = restitution
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

        if self.is_not_fixed():
            self._lower_bounds[self._name + "_linear_velocity"] = \
                np.array([-0.5] * 3)
            self._lower_bounds[self._name + "_angular_velocity"] = \
                np.array([-0.5] * 3)
            self._lower_bounds[self._name + "_mass"] = \
                np.array([0])

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

        if self.is_not_fixed():
            self._upper_bounds[self._name + "_linear_velocity"] = \
                np.array([0.5] * 3)
            self._upper_bounds[self._name + "_angular_velocity"] = \
                np.array([0.5] * 3)
            self._upper_bounds[self._name + "_mass"] = \
                np.array([0.2])

        self._state_variable_names = []

        if self.is_not_fixed():
            self._state_variable_names = ['type', 'position',
                                           'orientation', 'linear_velocity',
                                           'angular_velocity', 'mass',
                                           'size', 'color']
        else:
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

    def get_initial_position(self):
        return self._initial_position

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

    def get_recreation_params(self):
        raise NotImplementedError("the define type id function "
                                  "is not defined yet")

    def _init_object(self):
        for pybullet_client_id in self._pybullet_client_ids:
            shape_id, block_id =\
                self._create_object(pybullet_client_id)
            self._block_ids.append(block_id)
            self._shape_ids.append(shape_id)
        self._set_color(self._color)
        self._set_lateral_friction(self._lateral_friction)
        self._set_restitution(self._restitution)
        self._set_spinning_friction(self._spinning_friction)
        return

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
        elif variable_name == 'linear_velocity':
            linear_velocity, angular_velocity = pybullet.getBaseVelocity(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0])
            return linear_velocity

        elif variable_name == 'angular_velocity':
            linear_velocity, angular_velocity = pybullet.getBaseVelocity(
                self._block_ids[0],
                physicsClientId=self._pybullet_client_ids[0])
            return angular_velocity

        elif variable_name == 'mass':
            return self._mass
        elif variable_name == 'size':
            return self._size
        elif variable_name == 'color':
            return self._color

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
                                        color, 1),
                                       physicsClientId=
                                       self._pybullet_client_ids[i]
                                       )
        return

    def _set_lateral_friction(self, lateral_friction):
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(bodyUniqueId=self._block_ids[i],
                                    linkIndex=-1,
                                    lateralFriction=lateral_friction,
                                    physicsClientId=
                                    self._pybullet_client_ids[i]
                                    )

    def _set_restitution(self,restitution):
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(bodyUniqueId=self._block_ids[i],
                                    linkIndex=-1,
                                    restitution=restitution,
                                    physicsClientId=
                                    self._pybullet_client_ids[i]
                                    )

    def _set_spinning_friction(self, spinning_friction):
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(bodyUniqueId=self._block_ids[i],
                                    linkIndex=-1,
                                    spinningFriction=spinning_friction,
                                    physicsClientId=
                                    self._pybullet_client_ids[i]
                                    )

    def _set_velocities(self):
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.resetBaseVelocity(self._block_ids[i],
                                       self._initial_linear_velocity,
                                       self._initial_angular_velocity,
                                       physicsClientId=
                                       self._pybullet_client_ids[i])

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

            if self.is_not_fixed():
                linear_velocity, angular_velocity = \
                    pybullet.getBaseVelocity(self._block_ids[0],
                                             physicsClientId=
                                             self._pybullet_client_ids[0])
                state["linear_velocity"] = np.array(linear_velocity)
                state["angular_velocity"] = np.array(angular_velocity)
                state["mass"] = self._mass
        elif state_type == 'list':
            state = []

            position, orientation = pybullet.\
                getBasePositionAndOrientation(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
            if self.is_not_fixed():
                linear_velocity, angular_velocity = pybullet.\
                    getBaseVelocity(
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
                elif name == 'linear_velocity':
                    state.extend(linear_velocity)
                elif name == 'angular_velocity':
                    state.extend(angular_velocity)
                elif name == 'mass':
                    state.append(self._mass)
                elif name == 'size':
                    state.extend(self._size)
                elif name == 'color':
                    state.extend(self._color)
        return state

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
        if 'mass' in interventions_dict:
            self._mass = interventions_dict['mass']
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
        elif 'mass' in interventions_dict:
            for i in range(0, len(self._pybullet_client_ids)):
                pybullet.changeDynamics(
                    self._block_ids[i], -1, mass=self._mass,
                    physicsClientId=
                    self._pybullet_client_ids[i])

        if 'color' in interventions_dict:
            self._color = interventions_dict['color']
            self._set_color(self._color)
        if ('linear_velocity' in interventions_dict) ^ \
                ('angular_velocity' in interventions_dict):
            for i in range(0, len(self._pybullet_client_ids)):
                linear_velocity, angular_velocity = \
                    pybullet.getBaseVelocity(
                    self._block_ids[i],
                    physicsClientId=
                    self._pybullet_client_ids[i])
        if 'linear_velocity' in interventions_dict:
            linear_velocity = interventions_dict['linear_velocity']
        if 'angular_velocity' in interventions_dict:
            angular_velocity = interventions_dict['angular_velocity']
        if 'angular_velocity' in interventions_dict or 'linear_velocity' in \
                interventions_dict:
            for i in range(0, len(self._pybullet_client_ids)):
                pybullet.resetBaseVelocity(self._block_ids[i],
                                           linear_velocity,
                                           angular_velocity,
                                           physicsClientId=
                                           self._pybullet_client_ids[i])
        return

    def get_state_variable_names(self):
        """

        :return:
        """
        return self._state_variable_names

    def is_not_fixed(self):
        """

        :return:
        """
        return self._not_fixed

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
        #should be the same in both
        return pybullet.getAABB(self._block_ids[0],
                                physicsClientId=
                                self._pybullet_client_ids[0]
                                )

    def get_vertices(self):
        """

        :return:
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
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

    def world_to_cube_r_matrix(self):
        """

        :return:
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
        #TODO: double check if its not the inverse
        return get_transformation_matrix(position, orientation)

    def get_rotation_matrix(self):
        """

        :return:
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
        #TODO: double check if its not the inverse
        return get_rotation_matrix(orientation)

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


class Cuboid(RigidObject):
    def __init__(
        self,
        pybullet_client_ids, name,
        size=np.array([0.065, 0.065, 0.065]),
        initial_position=np.array([0.0, 0.0, 0.0425]),
        initial_orientation=np.array([0, 0, 0, 1]),
        mass=0.08,
        color=np.array([1, 0, 0]),
        initial_linear_velocity=np.array([0, 0, 0]),
        initial_angular_velocity=np.array([0, 0, 0]),
        lateral_friction=1,
        spinning_friction=0.001,
        restitution=0
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
        super(Cuboid, self).__init__(pybullet_client_ids=pybullet_client_ids,
                                     name=name,
                                     size=size,
                                     initial_position=initial_position,
                                     initial_orientation=initial_orientation,
                                     mass=mass,
                                     color=color,
                                     fixed_bool=False,
                                     lateral_friction=lateral_friction,
                                     spinning_friction=spinning_friction,
                                     restitution=restitution,
                                     initial_linear_velocity=
                                     initial_linear_velocity,
                                     initial_angular_velocity=
                                     initial_angular_velocity)

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(self._size) / 2,
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            basePosition=self._initial_position,
            baseOrientation=self._initial_orientation,
            baseMass=self._mass,
            physicsClientId=pybullet_client_id
        )
        return shape_id, block_id

    def _define_type_id(self):
        self._type_id = 10
        return

    def get_recreation_params(self):
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['size'] = self._size
        linear_velocity, angular_velocity = \
            pybullet.getBaseVelocity(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        recreation_params['initial_position'] = position
        recreation_params['initial_orientation'] = orientation
        recreation_params['mass'] = self._mass
        recreation_params['color'] = self._color
        recreation_params['lateral_friction'] = self._lateral_friction
        recreation_params['spinning_friction'] = self._spinning_friction
        recreation_params['restitution'] = self._restitution
        recreation_params['initial_linear_velocity'] = \
            linear_velocity
        recreation_params['initial_angular_velocity'] = \
            angular_velocity
        return copy.deepcopy(recreation_params)


class MeshObject(RigidObject):
    def __init__(
        self,
        pybullet_clients,
        name, filename,
        scale=np.array([0.01, 0.01, 0.01]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        mass=0.08, color=np.array([1, 0, 0])
    ):
        """

        :param pybullet_client:
        :param name:
        :param filename:
        :param scale:
        :param position:
        :param orientation:
        :param mass:
        :param color:
        """
        #TODO: intervene on friction as well
        self.__type_id = 0
        self.__mass = mass
        self.__scale = scale
        self.__filename = filename
        self.__not_fixed = True
        self.__color = color
        self.__block_ids = []
        self.__shape_ids = []
        for pybullet_client in pybullet_clients:
            __shape_id = pybullet_client.createCollisionShape(
                shapeType=pybullet_client.GEOM_MESH,
                meshScale=self.__scale,
                fileName=self.__filename)
            __block_id = pybullet_client.createMultiBody(
                baseCollisionShapeIndex=__shape_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=mass
            )
            self.__block_ids.append(__block_id)
            self.__shape_ids.append(__shape_id)
        super(MeshObject, self).__init__(pybullet_clients, name,
                                         self.__block_ids)
        for i in range(0, len(self._pybullet_clients)):
            self._pybullet_clients[i].changeVisualShape(self.__block_ids[i],
                                                        -1,
                                                        rgbaColor=np.append(
                                                            self.__color, 1))
            self._pybullet_clients[i].changeDynamics(
                bodyUniqueId=self.__block_ids[i],
                linkIndex=-1,
                restitution=0,
                lateralFriction=1,
                spinningFriction=0.001
            )
        #specifying bounds
        self.__lower_bounds = dict()
        self.__upper_bounds = dict()
        self.__lower_bounds[self._name + "_type"] = np.array([0])
        self.__lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.__lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.__lower_bounds[self._name + "_linear_velocity"] = \
            np.array([-0.5] * 3)
        self.__lower_bounds[self._name + "_angular_velocity"] = \
            np.array([-0.5] * 3)
        self.__lower_bounds[self._name + "_mass"] = \
            np.array([0])
        self.__lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self.__lower_bounds[self._name + "_color"] = \
            np.array([0]*3)

        self.__upper_bounds[self._name + "_type"] = np.array([10])
        self.__upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self.__upper_bounds[self._name + "_linear_velocity"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_angular_velocity"] = \
            np.array([0.5] * 3)
        self.__upper_bounds[self._name + "_mass"] = \
            np.array([0.2])
        self.__upper_bounds[self._name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.__upper_bounds[self._name + "_color"] = \
            np.array([1]*3)
        self.__state_variable_names = ['type', 'position',
                                       'orientation', 'linear_velocity',
                                       'angular_velocity', 'mass',
                                       'size', 'color']
        self.__state_variable_sizes = []
        self.__state_size = 0
        for state_variable_name in self.__state_variable_names:
            self.__state_variable_sizes.append(
                self.__upper_bounds[self._name + "_" +
                                    state_variable_name].shape[0])
            self.__state_size += self.__state_variable_sizes[-1]
        self.__volume = None
        self.__size = None
        self.__set_volume()
        self.__initial_position = position
        self.__initial_orientation = orientation

    def get_initial_position(self):
        return self.__initial_position

    def get_initial_orientation(self):
        return self.__initial_orientation

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        #form dict first
        new_state_dict = dict()
        current_state = self.get_state()
        start = 0
        for i in range(len(self.__state_variable_sizes)):
            end = start + self.__state_variable_sizes[i]
            if not np.all(current_state[self.__state_variable_names[i]] ==
                          new_state[start:end]):
                if end == start + 1:
                    new_state_dict[self.__state_variable_names[i]] = new_state[start:end][0]
                else:
                    new_state_dict[self.__state_variable_names[i]] = new_state[start:end]
            start = end
        self.set_state(new_state_dict)
        return

    def set_state(self, state_dict):
        """

        :param state_dict:
        :return:
        """
        if 'position' not in state_dict or 'orientation' not in state_dict:
            position, orientation = self._pybullet_clients[0].\
                getBasePositionAndOrientation(
                self.__block_ids[0]
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'mass' in state_dict:
            self.__mass = state_dict['mass'][0]
        elif 'position' in state_dict or 'orientation' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], position, orientation
                )
        elif 'mass' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeDynamics(
                    self.__block_ids[i], -1, mass=self.__mass)

        if 'color' in  state_dict:
            self.__color = state_dict['color']
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(
                    self.__block_ids[i], -1, rgbaColor=np.append(
                        state_dict['color'], 1))
        if ('linear_velocity' in state_dict) ^ \
                ('angular_velocity' in state_dict):
            for i in range(0, len(self._pybullet_clients)):
                linear_velocity, angular_velocity = \
                    self._pybullet_clients[i].getBaseVelocity(
                    self.__block_ids[i])
        if 'linear_velocity' in state_dict:
            linear_velocity = state_dict['linear_velocity']
        if 'angular_velocity' in state_dict:
            angular_velocity = state_dict['angular_velocity']
        if 'angular_velocity' in state_dict or 'linear_velocity' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBaseVelocity(self.__block_ids[i],
                                                            linear_velocity,
                                                            angular_velocity)
        return

    def do_intervention(self, variable_name, variable_value):
        #TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            for i in range(0, len(self._pybullet_clients)):
                position, orientation = \
                    self._pybullet_clients[i].getBasePositionAndOrientation(
                    self.__block_ids[i]
                )
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], variable_value, orientation
                )
        elif variable_name == 'orientation':
            for i in range(0, len(self._pybullet_clients)):
                position, orientation = \
                    self._pybullet_clients[i].getBasePositionAndOrientation(
                    self.__block_ids[i]
                )
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], position, variable_value
                )
        elif variable_name == 'mass':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].\
                    changeDynamics(self.__block_ids[i], -1, mass=variable_value)
                self.__mass = variable_value
        elif variable_name == 'color':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(self.__block_ids[i], -1,
                                                            rgbaColor=np.append
                                                            (variable_value, 1))
            self.__color = variable_value
        elif variable_name == 'linear_velocity':
            for i in range(0, len(self._pybullet_clients)):
                _, angular_velocity = self._pybullet_clients[i].getBaseVelocity(
                    self.__block_ids[i])
                self._pybullet_clients[i].resetBaseVelocity(self.__block_ids[i],
                                                        variable_value,
                                                        angular_velocity)
        elif variable_name == 'angular_velocity':
            for i in range(0, len(self._pybullet_clients)):
                linear_velocity, _ = self._pybullet_clients[i].getBaseVelocity(
                    self.__block_ids[i])
                self._pybullet_clients[i].resetBaseVelocity(self.__block_ids[i],
                                                            linear_velocity,
                                                            variable_value)
        else:
            raise Exception("The variable passed cant intervene on")
        #TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
        """
        if state_type == 'dict':
            state = dict()
            position, orientation = \
                self._pybullet_clients[0].getBasePositionAndOrientation(
                self.__block_ids[0]
            )
            state["type"] = self.__type_id
            state["position"] = np.array(position)
            state["orientation"] = np.array(orientation)
            linear_velocity, angular_velocity = \
                self._pybullet_clients[0].getBaseVelocity(self.__block_ids[0])
            state["linear_velocity"] = np.array(linear_velocity)
            state["angular_velocity"] = np.array(angular_velocity)
            state["mass"] = self.__mass
            state["size"] = self.__size
            state["color"] = self.__color
        elif state_type == 'list':
            state = []
            position, orientation = self._pybullet_clients[0].\
                getBasePositionAndOrientation(
                self.__block_ids[0]
            )
            linear_velocity, angular_velocity = self._pybullet_clients[0].\
                getBaseVelocity(
                self.__block_ids[0])
            for name in self.__state_variable_names:
                if name == 'type':
                    state.append(self.__type_id)
                elif name == 'position':
                    state.extend(position)
                elif name == 'orientation':
                    state.extend(orientation)
                elif name == 'linear_velocity':
                    state.extend(linear_velocity)
                elif name == 'angular_velocity':
                    state.extend(angular_velocity)
                elif name == 'mass':
                    state.append(self.__mass)
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
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            return position

        elif variable_name == 'orientation':
            position, orientation = self._pybullet_clients[0].\
                getBasePositionAndOrientation(
                self.__block_ids[0]
            )
            return orientation
        elif variable_name == 'linear_velocity':
            linear_velocity, angular_velocity = self._pybullet_clients[0].\
                getBaseVelocity(
                self.__block_ids[0])
            return linear_velocity

        elif variable_name == 'angular_velocity':
            linear_velocity, angular_velocity = self._pybullet_clients[0].\
                getBaseVelocity(
                self.__block_ids[0])
            return angular_velocity

        elif variable_name == 'mass':
            return self.__mass
        elif variable_name == 'size':
            return self.__size
        elif variable_name == 'color':
            return self.__color
        else:
            raise Exception("variable name is not supported")

    def get_state_variable_names(self):
        """

        :return:
        """
        return self.__state_variable_names

    def is_not_fixed(self):
        """

        :return:
        """
        return self.__not_fixed

    def get_bounds(self):
        """

        :return:
        """
        return self.__lower_bounds, self.__upper_bounds

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
        return

    def __set_volume(self):
        """

        :return:
        """
        bb = self.get_bounding_box()
        self.__size = np.array([(bb[1][0] - bb[0][0]), (bb[1][1] - bb[0][1]),
                                (bb[1][2] - bb[0][2])])
        self.__volume = self.__size[0] * self.__size[1] * self.__size[2]
        return

    def get_volume(self):
        """

        :return:
        """
        return self.__volume

    def get_size(self):
        """

        :return:
        """
        return self.__size


class StaticCuboid(RigidObject):
    # TODO: implement get bounding box and get area
    def __init__(
        self,
        pybullet_clients,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        color=np.array([0, 0, 0])
    ):
        """

        :param pybullet_clients:
        :param name:
        :param size:
        :param position:
        :param orientation:
        :param color:
        """
        #TODO: intervene on friction as well
        self.type_id = 10 #TODO: static objects ids start from 10
        self.size = size
        self.not_fixed = False
        self.__color = color

        self.__block_ids = []
        self.__shape_ids = []
        for pybullet_client in pybullet_clients:
            __shape_id = pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(size) / 2)
            __block_id = pybullet_client.createMultiBody(
                baseCollisionShapeIndex=__shape_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=0
            )
            self.__block_ids.append(__block_id)
            self.__shape_ids.append(__shape_id)
        super(StaticCuboid, self).__init__(pybullet_clients, name,
                                           self.__block_ids)
        for i in range(0, len(self._pybullet_clients)):
            self._pybullet_clients[i].changeVisualShape(self.__block_ids[i],
                                                        -1,
                                                        rgbaColor=np.append(
                                                            self.__color, 1))
            self._pybullet_clients[i].changeDynamics(
                bodyUniqueId=self.__block_ids[i],
                linkIndex=-1,
                restitution=0,
                lateralFriction=1,
                spinningFriction=0.001
            )
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self._name + "_type"] = np.array([0])
        self.lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self.lower_bounds[self._name + "_color"] = \
            np.array([0]*3)

        self.upper_bounds[self._name + "_type"] = np.array([10])
        self.upper_bounds[self._name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self._name + "_orientation"] = \
            np.array([10] * 4)
        self.upper_bounds[self._name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.upper_bounds[self._name + "_color"] = \
            np.array([1]*3)
        self._state_variable_names = ['type', 'position',
                                      'orientation',
                                      'size', 'color']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self._name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]
        self.position = position
        self.orientation = orientation

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
        if 'orientation' in state_dict:
            self.orientation = state_dict['orientation']
        if 'size' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].removeBody(self.__block_ids[i])
            self.__block_ids = []
            self.__shape_ids = []
            for i in range(0, len(self._pybullet_clients)):
                __shape_id = self._pybullet_clients[i].createCollisionShape(
                    shapeType=pybullet.GEOM_BOX,
                    halfExtents=np.array(state_dict['size']) / 2)
                __block_id = self._pybullet_clients[i].createMultiBody(
                    baseCollisionShapeIndex=__shape_id,
                    basePosition=self.position,
                    baseOrientation=self.orientation,
                    baseMass=0
                )
                self.__block_ids.append(__block_id)
                self.__shape_ids.append(__shape_id)
            self.size = state_dict['size']
        elif 'position' in state_dict or 'orientation' in state_dict:
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.position, self.orientation
                )
        if 'color' in state_dict:
            self.color = state_dict['color']
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(
                    self.__block_ids[i], -1, rgbaColor=
                    np.append(state_dict['color'], 1))
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        #TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            self.position = variable_value
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.position, self.orientation
                )
        elif variable_name == 'orientation':
            self.orientation = variable_value
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].resetBasePositionAndOrientation(
                    self.__block_ids[i], self.position, self.orientation
                )
        elif variable_name == 'size':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].removeBody(self.__block_ids[i])
            self.__block_ids = []
            self.__shape_ids = []
            for i in range(0, len(self._pybullet_clients)):
                __shape_id = self._pybullet_clients[i].createCollisionShape(
                    shapeType=pybullet.GEOM_BOX,
                    halfExtents=np.array(variable_value) / 2)
                __block_id = self._pybullet_clients[i].createMultiBody(
                    baseCollisionShapeIndex=__shape_id,
                    basePosition=self.position,
                    baseOrientation=self.orientation,
                    baseMass=0
                )

                self.__block_ids.append(__block_id)
                self.__shape_ids.append(__shape_id)
            self.size = variable_value
        elif variable_name == 'color':
            for i in range(0, len(self._pybullet_clients)):
                self._pybullet_clients[i].changeVisualShape(self.__block_ids[i], -1,
                                                            rgbaColor=np.append(variable_value, 1))
            self.color = variable_value
        #TODO: implement intervention on shape id itself
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type:
        :return:
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
        """

        :param variable_name:
        :return:
        """
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

    def get_state_variable_names(self):
        """

        :return:
        """
        return self._state_variable_names

    def is_not_fixed(self):
        """

        :return:
        """
        return self.not_fixed

    def get_bounds(self):
        """

        :return:
        """
        return self.lower_bounds, self.upper_bounds

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
                self.__block_ids[i], position, orientation
            )
        self.position = position
        self.orientation = orientation
        return







