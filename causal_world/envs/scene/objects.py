import pybullet
import numpy as np
from causal_world.utils.rotation_utils import rotate_points, \
    get_transformation_matrix, get_rotation_matrix, cyl2cart, cart2cyl, euler_to_quaternion
import copy
from causal_world.configs.world_constants import WorldConstants


class RigidObject(object):

    def __init__(self, pybullet_client_ids, name, size, initial_position,
                 initial_orientation, mass, color, lateral_friction,
                 spinning_friction, restitution, initial_linear_velocity,
                 initial_angular_velocity, fixed_bool):
        """
        This is the base class of any rigid object whether it is fixed or not.

        :param pybullet_client_ids: (list) specifies the pybullet client ids
                                           where this object will be in.
        :param name: (str) specifies the name of the object, needs to be unique.
        :param size: (float list) 3 dimensional list specifies the size
        :param initial_position: (float list) specifies the x,y,z position in
                                              the arena.
        :param initial_orientation: (float list) specifies the quaternion
                                                 orientation.
        :param mass: (float) specifies the mass of the object itself.
                             0 if fixed.
        :param color: (float list) specifies the RGB values of the object.
        :param lateral_friction: (float) specifies the lateral friction of the
                                         object.
        :param spinning_friction: (float) specifies the spinning friction of the
                                          object.
        :param restitution: (float) specifies the restitution of the object.
        :param initial_linear_velocity: (float list) specifies the velocity in
                                                     the x,y,z directions.
        :param initial_angular_velocity: (float list) specifies the velocity in
                                                      the yaw, roll, pitch values.
        :param fixed_bool: (bool) specifies if the object is fixed or not.
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
        self._lower_bounds[self._name + "_friction"] = \
            np.array([0])
        self._lower_bounds[self._name + "_size"] = \
            np.array([0.03, 0.03, 0.03])
        self._lower_bounds[self._name + "_color"] = \
            np.array([0] * 3)

        if self.is_not_fixed():
            self._lower_bounds[self._name + "_linear_velocity"] = \
                np.array([-0.5] * 3)
            self._lower_bounds[self._name + "_angular_velocity"] = \
                np.array([-np.pi] * 3)
            self._lower_bounds[self._name + "_mass"] = \
                np.array([0.02])
        #decision: type id is not normalized
        self._upper_bounds[self._name + "_type"] = \
            np.array([self._type_id])
        self._upper_bounds[self._name + "_friction"] = \
            np.array([10])
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

        if self.is_not_fixed():
            self._upper_bounds[self._name + "_linear_velocity"] = \
                np.array([0.5] * 3)
            self._upper_bounds[self._name + "_angular_velocity"] = \
                np.array([np.pi] * 3)
            self._upper_bounds[self._name + "_mass"] = \
                np.array([0.2])
        self._state_variable_names = []

        if self.is_not_fixed():
            self._state_variable_names = [
                'type', 'cartesian_position', 'cylindrical_position',
                'orientation', 'linear_velocity', 'angular_velocity', 'mass',
                'size', 'color', 'friction', 'type'
            ]
        else:
            self._state_variable_names = [
                'type', 'cartesian_position', 'cylindrical_position',
                'orientation', 'size', 'color', 'friction', 'type'
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

    def get_initial_position(self):
        """

        :return: (nd.array) initial position where the object was created.
        """
        return self._initial_position

    def _set_volume(self):
        """
        sets the volume based on the size of the object or otherwise.

        :return:
        """
        self._volume = self._size[0] * self._size[1] * self._size[2]
        return

    def _add_state_variables(self):
        """

        :return:
        """
        return

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

        :param pybullet_client_id: (int) pybullet client id to create the
                                         object in.
        :param kwargs: (params) parameters of the object to be created.
        :return:
        """
        raise NotImplementedError("the creation function is not defined "
                                  "yet")

    def _define_type_id(self):
        """
        defines the type id.

        :return:
        """
        raise NotImplementedError("the define type id function "
                                  "is not defined yet")

    def get_recreation_params(self):
        """
        gets the params that are needed to recreate the same
        object again.

        :return:
        """
        raise NotImplementedError("the define type id function "
                                  "is not defined yet")

    def _init_object(self):
        """
        initializes the object using this function.

        :return:
        """
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

        :param variable_name: (str) variable name to query about the object.
        :return: (nd.array, float or int) returns the corresponding value of
                                          the variable.
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
        elif variable_name == 'friction':
            return self._lateral_friction

    def reinit_object(self):
        """
        removes the object and reinitilaizes it again.

        :return:
        """
        self.remove()
        self._init_object()
        return

    def remove(self):
        """
        removes the object.

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

        :param color: (nd.array) the normalized RGB color, shape is (3,)
        :return:
        """
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeVisualShape(
                self._block_ids[i],
                -1,
                rgbaColor=np.append(color, 1),
                physicsClientId=self._pybullet_client_ids[i])
        return

    def _set_lateral_friction(self, lateral_friction):
        """

        :param lateral_friction: (float) specifies the lateral friction.
        :return:
        """
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(bodyUniqueId=self._block_ids[i],
                                    linkIndex=-1,
                                    lateralFriction=lateral_friction,
                                    physicsClientId=
                                    self._pybullet_client_ids[i]
                                    )

    def _set_restitution(self,restitution):
        """

        :param restitution: (float) specifies the restitution.
        :return:
        """
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(
                bodyUniqueId=self._block_ids[i],
                linkIndex=-1,
                restitution=restitution,
                physicsClientId=self._pybullet_client_ids[i])

    def _set_spinning_friction(self, spinning_friction):
        """

        :param spinning_friction: (float) specifies the spinning friction.
        :return:
        """
        for i in range(len(self._pybullet_client_ids)):
            pybullet.changeDynamics(
                bodyUniqueId=self._block_ids[i],
                linkIndex=-1,
                spinningFriction=spinning_friction,
                physicsClientId=self._pybullet_client_ids[i])

    def _set_velocities(self):
        """
        sets the velocities specified when constructing the object.

        :return:
        """
        for i in range(0, len(self._pybullet_client_ids)):
            pybullet.resetBaseVelocity(
                self._block_ids[i],
                self._initial_linear_velocity,
                self._initial_angular_velocity,
                physicsClientId=self._pybullet_client_ids[i])

    def set_pose(self, position, orientation):
        """
        :param position: (nd.array) specifies the cartesian position of the
                                    object.
        :param orientation: (nd.array) specifies the quaternion orientation
                                       of the object.
        :return:
        """
        for i in range(0, len(self._pybullet_client_ids)):
            position[-1] += WorldConstants.FLOOR_HEIGHT
            pybullet.resetBasePositionAndOrientation(
                self._block_ids[i],
                position,
                orientation,
                physicsClientId=self._pybullet_client_ids[i])
        return

    def get_state(self, state_type='dict'):
        """

        :param state_type: 'list' or 'dict'.
        :return: (dict or list) specifies the state of the object.
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
            state["friction"] = self._lateral_friction

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
            position = np.array(position)
            position[-1] -= WorldConstants.FLOOR_HEIGHT
            if self.is_not_fixed():
                linear_velocity, angular_velocity = pybullet.\
                    getBaseVelocity(
                    self._block_ids[0],
                    physicsClientId=
                    self._pybullet_client_ids[0])
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self._type_id)
                elif name == 'cartesian_position':
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
                elif name == 'friction':
                    state.append(self._lateral_friction)
        return state

    def set_full_state(self, new_state):
        """

        :param new_state: (list) specifies the state of the object to be set.
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
                                          performed on the various variables.
        :return:
        """
        #TODO: Add frictions to apply interventions
        if 'cylindrical_position' in interventions_dict:
            interventions_dict['cartesian_position'] = cyl2cart(
                interventions_dict['cylindrical_position'])
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
        if 'mass' in interventions_dict:
            self._mass = interventions_dict['mass']
        if 'friction' in interventions_dict:
            self._lateral_friction = interventions_dict['friction']
        if 'size' in interventions_dict:
            self._size = interventions_dict['size']
            self._set_volume()
            self.reinit_object()
        elif 'mass' in interventions_dict:
            for i in range(0, len(self._pybullet_client_ids)):
                pybullet.changeDynamics(
                    self._block_ids[i],
                    -1,
                    mass=self._mass,
                    physicsClientId=self._pybullet_client_ids[i])
        elif 'friction' in interventions_dict:
            self._set_lateral_friction(self._lateral_friction)

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
                pybullet.resetBaseVelocity(
                    self._block_ids[i],
                    linear_velocity,
                    angular_velocity,
                    physicsClientId=self._pybullet_client_ids[i])
        return

    def get_state_variable_names(self):
        """

        :return: (list) returns the state variable names.
        """
        return self._state_variable_names

    def is_not_fixed(self):
        """

        :return: (bool) true if its not fixed object.
        """
        return self._not_fixed

    def get_bounds(self):
        """

        :return: (tuple) first position of the tuple is the lower bound of the
                         bounding box of the object and second position of the
                         tuple is the upper bound of the bounding box.
        """
        return self._lower_bounds, self._upper_bounds

    def get_state_size(self):
        """

        :return: (int) specifies how large is the state of the object.
        """
        return self._state_size

    def get_bounding_box(self):
        """

        :return: (nd.array) first position of the array is the lower bound of the
                            bounding box of the object and second position of the
                            array is the upper bound of the bounding box.
        """
        #should be the same in both
        bb = pybullet.getAABB(self._block_ids[0],
                              physicsClientId=self._pybullet_client_ids[0])
        bb = np.array(bb)
        bb[0][-1] -= WorldConstants.FLOOR_HEIGHT
        bb[1][-1] -= WorldConstants.FLOOR_HEIGHT
        return bb

    def get_vertices(self):
        """

        :return: (nd.array) specifies the current vertices of the object.
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        vertices = [[1, 1, -1, 1], [1, -1, -1, 1], [-1, 1, -1, 1], [-1, -1, -1, 1],
                    [1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1]]
        temp_size = np.array([self._size[0], self._size[1], self._size[2], 2])
        vertices = [(point * temp_size / 2.0) for point in vertices]
        return rotate_points(np.array(vertices), orientation, position)

    def world_to_cube_r_matrix(self):
        """

        :return: (nd.array) returns the transformation matrix of the object.
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        #TODO: double check if its not the inverse
        return get_transformation_matrix(position, orientation)

    def get_rotation_matrix(self):
        """

        :return: (nd.array) returns the rotation matrix of the object.
        """
        position, orientation = pybullet.\
            getBasePositionAndOrientation(
            self._block_ids[0],
            physicsClientId=
            self._pybullet_client_ids[0]
        )
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        #TODO: double check if its not the inverse
        return get_rotation_matrix(orientation)

    def get_size(self):
        """

        :return: (nd.array) returns the size of the object.
        """
        return self._size

    def get_volume(self):
        """

        :return: (nd.array) returns the volume of the object.
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


class Cuboid(RigidObject):

    def __init__(
            self,
            pybullet_client_ids,
            name,
            size=np.array([0.065, 0.065, 0.065]),
            initial_position=np.array([0.0, 0.0, 0.0425]),
            initial_orientation=np.array([0, 0, 0, 1]),
            mass=0.08,
            color=np.array([1, 0, 0]),
            initial_linear_velocity=np.array([0, 0, 0]),
            initial_angular_velocity=np.array([0, 0, 0]),
            lateral_friction=1,
    ):
        """
        This specifies the moving cuboid object in the arena.

        :param pybullet_client_ids: (list) specifies the pybullet client ids.
        :param name: (str) specifies the name of the object.
        :param size: (list float) specifies the size in the three directions.
        :param initial_position: (list float) specifies the position in x,y,z.
        :param initial_orientation: (list float) specifies the quaternion of
                                                 the object.
        :param mass: (float) specifies the mass of the object.
        :param color: (list float) specifies the RGB values of the cuboid.
        :param initial_linear_velocity: (list float) specifies the initial
                                                     linear velocity vx, vy, vz.
        :param initial_angular_velocity: (list float) specifies the initial
                                                      angular velocities.
        :param lateral_friction: (float) specifies the lateral friction.
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
                                     spinning_friction=0.001,
                                     restitution=0,
                                     initial_linear_velocity=
                                     initial_linear_velocity,
                                     initial_angular_velocity=
                                     initial_angular_velocity)

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

        :param pybullet_client_id: (int) corresponding pybullet client to create
                                         the object in.
        :param kwargs: (params) parameters for the object creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(self._size) / 2,
            physicsClientId=pybullet_client_id)
        position = np.array(self._initial_position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        block_id = pybullet.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            basePosition=position,
            baseOrientation=self._initial_orientation,
            baseMass=self._mass,
            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the type id.

        :return:
        """
        self._type_id = 1
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the object.
        """
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
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        recreation_params['initial_position'] = position
        recreation_params['initial_orientation'] = orientation
        recreation_params['mass'] = self._mass
        recreation_params['color'] = self._color
        recreation_params['lateral_friction'] = self._lateral_friction
        recreation_params['initial_linear_velocity'] = \
            linear_velocity
        recreation_params['initial_angular_velocity'] = \
            angular_velocity
        return copy.deepcopy(recreation_params)


class StaticCuboid(RigidObject):

    def __init__(self,
                 pybullet_client_ids,
                 name,
                 size=np.array([0.065, 0.065, 0.065]),
                 position=np.array([0.0, 0.0, 0.0425]),
                 orientation=np.array([0, 0, 0, 1]),
                 color=np.array([1, 0, 0]),
                 lateral_friction=1):
        """

        :param pybullet_client_ids: (list) specifies the pybullet clients.
        :param name: (str) specifies the name of the object.
        :param size: (list float) specifies the size in the three directions.
        :param position: (list float) specifies the position in x,y,z.
        :param orientation: (list float) specifies the quaternion of
                                                        the object.
        :param color: (list float) specifies the RGB values of the cuboid.
        :param lateral_friction: (float) specifies the lateral friction.
        """
        #TODO: intervene on friction as well
        super(StaticCuboid, self).__init__(pybullet_client_ids=
                                           pybullet_client_ids,
                                           name=name,
                                           size=size,
                                           initial_position=position,
                                           initial_orientation=orientation,
                                           mass=0,
                                           color=color,
                                           fixed_bool=True,
                                           lateral_friction=lateral_friction,
                                           spinning_friction=0.001,
                                           restitution=0,
                                           initial_linear_velocity=
                                           [0, 0, 0],
                                           initial_angular_velocity=
                                           [0, 0, 0])

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

         :param pybullet_client_id: (int) corresponding pybullet client to create
                                         the object in.
        :param kwargs: (params) parameters for the object creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        position = np.array(self._initial_position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=np.array(self._size) / 2,
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            basePosition=position,
            baseOrientation=self._initial_orientation,
            baseMass=self._mass,
            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the type id of the object.

        :return:
        """
        self._type_id = 10
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the object.
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
        recreation_params['lateral_friction'] = self._lateral_friction
        return copy.deepcopy(recreation_params)


class MeshObject(RigidObject):

    def __init__(self,
                 pybullet_client_ids,
                 name,
                 filename,
                 scale=np.array([0.01, 0.01, 0.01]),
                 initial_position=np.array([0.0, 0.0, 0.0425]),
                 initial_orientation=np.array([0, 0, 0, 1]),
                 color=np.array([1, 0, 0]),
                 mass=0.08,
                 initial_linear_velocity=np.array([0, 0, 0]),
                 initial_angular_velocity=np.array([0, 0, 0]),
                 lateral_friction=1):
        """

        :param pybullet_client_ids: (list) specifies the pybullet clients.
        :param name: (str) specifies the name of the object.
        :param filename: (str) specifies the name of the file itself.
        :param scale: (list float) specifies the scale of the mesh object.
        :param initial_position: (list float) specifies the positions in x,y,z
        :param initial_orientation: (list float) specifies the quaternion of the object.
        :param color: (list float) specifies the RGB values.
        :param mass: (float) specifies the object mass.
        :param initial_linear_velocity: (list float) specifies the velocity in vx, vy, vz.
        :param initial_angular_velocity: (list float) specifies the velocity in yaw, roll, pitch.
        :param lateral_friction: (float) specifies the lateral friction.
        """
        #TODO: intervene on friction as well
        self._scale = scale
        self._filename = filename
        super(MeshObject,
              self).__init__(pybullet_client_ids=pybullet_client_ids,
                             name=name,
                             size=[0, 0, 0],
                             initial_position=initial_position,
                             initial_orientation=initial_orientation,
                             mass=mass,
                             color=color,
                             fixed_bool=False,
                             lateral_friction=lateral_friction,
                             spinning_friction=0.001,
                             restitution=0,
                             initial_linear_velocity=initial_linear_velocity,
                             initial_angular_velocity=initial_angular_velocity)
        bb = self.get_bounding_box()
        self._size = np.array([(bb[1][0] - bb[0][0]), (bb[1][1] - bb[0][1]),
                               (bb[1][2] - bb[0][2])])

    def _create_object(self, pybullet_client_id,
                       **kwargs):
        """

         :param pybullet_client_id: (int) corresponding pybullet client to create
                                         the object in.
        :param kwargs: (params) parameters for the object creation.

        :return: (tuple) the first position specifies the shape_id and the
                         second specifies the block id for pybullet.
        """
        position = np.array(self._initial_position)
        position[-1] += WorldConstants.FLOOR_HEIGHT
        shape_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            meshScale=self._scale,
            fileName=self._filename,
            physicsClientId=pybullet_client_id)
        block_id = pybullet.createMultiBody(
            baseCollisionShapeIndex=shape_id,
            basePosition=position,
            baseOrientation=self._initial_orientation,
            baseMass=self._mass,
            physicsClientId=pybullet_client_id)
        return shape_id, block_id

    def _define_type_id(self):
        """
        Sets the type id of the object.

        :return:
        """
        self._type_id = 2
        return

    def get_recreation_params(self):
        """

        :return: (dict) the creation parameters needed to recreate the object.
        """
        recreation_params = dict()
        recreation_params['name'] = self._name
        recreation_params['filename'] = self._filename
        recreation_params['scale'] = self._scale
        position, orientation = pybullet. \
            getBasePositionAndOrientation(self._block_ids[0],
                                          physicsClientId=
                                          self._pybullet_client_ids[0])
        linear_velocity, angular_velocity = \
            pybullet.getBaseVelocity(
                self._block_ids[0],
                physicsClientId=
                self._pybullet_client_ids[0])
        position = np.array(position)
        position[-1] -= WorldConstants.FLOOR_HEIGHT
        recreation_params['initial_position'] = position
        recreation_params['initial_orientation'] = orientation
        recreation_params['mass'] = self._mass
        recreation_params['initial_linear_velocity'] = linear_velocity
        recreation_params['initial_angular_velocity'] = angular_velocity
        recreation_params['color'] = self._color
        recreation_params['lateral_friction'] = self._lateral_friction
        return copy.deepcopy(recreation_params)
