import pybullet
import numpy as np
from causal_rl_bench.utils.rotation_utils import rotate_points, \
    get_transformation_matrix, get_rotation_matrix


class RigidObject(object):
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

    def is_not_fixed(self):
        """

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
        return self._pybullet_client.getAABB(self._block_id)

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


class Cuboid(RigidObject):
    def __init__(
        self,
        pybullet_client,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        mass=0.08, color=np.array([1, 0, 0])
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
        self.__type_id = 0
        self.__mass = mass
        self.__size = size
        self.__not_fixed = True
        self.__color = color
        self.__shape_id = pybullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, halfExtents=np.array(size)/2)
        self.__block_id = pybullet_client.createMultiBody(
            baseCollisionShapeIndex=self.__shape_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=mass
        )
        super(Cuboid, self).__init__(pybullet_client, name, self.__block_id)
        self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                rgbaColor=np.append(
                                                   self.__color, 1))
        self._pybullet_client.changeDynamics(
            bodyUniqueId=self.__block_id,
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
            np.array([0.065, 0.065, 0.065])
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
        self.__set_volume()

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
            position, orientation = self._pybullet_client.\
                getBasePositionAndOrientation(
                self.__block_id
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'mass' in state_dict:
            self.__mass = state_dict['mass'][0]
        if 'size' in state_dict:
            self._pybullet_client.removeBody(self.__block_id)
            self.__shape_id = self._pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(state_dict['size']) / 2)
            self.__block_id = self._pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.__shape_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=self.__mass
            )
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=
                                                   np.append(self.__color, 1))
            self.__size = state_dict['size']
            self.__set_volume()
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, position, orientation
            )
        elif 'mass' in state_dict:
            self._pybullet_client.changeDynamics(self.__block_id, -1,
                                                 mass=self.__mass)

        if 'color' in  state_dict:
            self.__color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=
                                                   np.append(state_dict['color'], 1))
        if ('linear_velocity' in state_dict) ^ \
                ('angular_velocity' in state_dict):
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
        if 'linear_velocity' in state_dict:
            linear_velocity = state_dict['linear_velocity']
        if 'angular_velocity' in state_dict:
            angular_velocity = state_dict['angular_velocity']
        if 'angular_velocity' in state_dict or 'linear_velocity' in state_dict:
            self._pybullet_client.resetBaseVelocity(self.__block_id,
                                                    linear_velocity,
                                                    angular_velocity)
        position, orientation = \
            self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
        return

    def do_intervention(self, variable_name, variable_value):
        """

        :param variable_name:
        :param variable_value:
        :return:
        """
        #TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, variable_value, orientation
            )
        elif variable_name == 'orientation':
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, position, variable_value
            )
        elif variable_name == 'mass':
            self._pybullet_client.changeDynamics(self.__block_id, -1, mass=variable_value)
            self.__mass = variable_value
        elif variable_name == 'size':
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            self._pybullet_client.removeBody(self.__block_id)
            self.__shape_id = self._pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX, halfExtents=np.array(variable_value) / 2)
            self.__block_id = self._pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.__shape_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=self.__mass
            )
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=np.append(self.__color, 1))
            self.__size = variable_value
            self.__set_volume()
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=np.append(variable_value, 1))
            self.__color = variable_value
        elif variable_name == 'linear_velocity':
            _, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            self._pybullet_client.resetBaseVelocity(self.__block_id,
                                                    variable_value,
                                                    angular_velocity)
        elif variable_name == 'angular_velocity':
            linear_velocity, _ = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            self._pybullet_client.resetBaseVelocity(self.__block_id,
                                                    linear_velocity,
                                                    variable_value)
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
                self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            state["type"] = self.__type_id
            state["position"] = np.array(position)
            state["orientation"] = np.array(orientation)
            linear_velocity, angular_velocity = \
                self._pybullet_client.getBaseVelocity(self.__block_id)
            state["linear_velocity"] = np.array(linear_velocity)
            state["angular_velocity"] = np.array(angular_velocity)
            state["mass"] = self.__mass
            state["size"] = self.__size
            state["color"] = self.__color
        elif state_type == 'list':
            state = []
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
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
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            return orientation
        elif variable_name == 'linear_velocity':
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            return linear_velocity

        elif variable_name == 'angular_velocity':
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
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
        self._pybullet_client.resetBasePositionAndOrientation(
            self.__block_id, position, orientation
        )
        return

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

    def get_vertices(self):
        """

        :return:
        """
        position, orientation = self._pybullet_client.getBasePositionAndOrientation(
            self.__block_id
        )
        vertices = [[1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1]]
        vertices = [position + (point * self.__size / 2)
                    for point in vertices]
        return rotate_points(np.array(vertices), orientation)

    def world_to_cube_r_matrix(self):
        """

        :return:
        """
        position, orientation = self._pybullet_client.getBasePositionAndOrientation(
            self.__block_id
        )
        #TODO: double check if its not the inverse
        return get_transformation_matrix(position, orientation)

    def get_rotation_matrix(self):
        """

        :return:
        """
        position, orientation = self._pybullet_client.getBasePositionAndOrientation(
            self.__block_id
        )
        #TODO: double check if its not the inverse
        return get_rotation_matrix(orientation)

    def get_size(self):
        """

        :return:
        """
        return self.__size


class MeshObject(RigidObject):
    def __init__(
        self,
        pybullet_client,
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
        self.__shape_id = pybullet_client.createCollisionShape(
            shapeType=pybullet_client.GEOM_MESH,
            meshScale=self.__scale,
            fileName=self.__filename)
        self.__block_id = pybullet_client.createMultiBody(
            baseCollisionShapeIndex=self.__shape_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=mass
        )
        super(MeshObject, self).__init__(pybullet_client, name, self.__block_id)
        self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                rgbaColor=np.append(
                                                   self.__color, 1))
        self._pybullet_client.changeDynamics(
            bodyUniqueId=self.__block_id,
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
            np.array([0.065, 0.065, 0.065])
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
            position, orientation = self._pybullet_client.\
                getBasePositionAndOrientation(
                self.__block_id
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'mass' in state_dict:
            self.__mass = state_dict['mass'][0]
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, position, orientation
            )
        elif 'mass' in state_dict:
            self._pybullet_client.changeDynamics(self.__block_id, -1,
                                                 mass=self.__mass)

        if 'color' in  state_dict:
            self.__color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=
                                                   np.append(state_dict['color'], 1))
        if ('linear_velocity' in state_dict) ^ \
                ('angular_velocity' in state_dict):
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
        if 'linear_velocity' in state_dict:
            linear_velocity = state_dict['linear_velocity']
        if 'angular_velocity' in state_dict:
            angular_velocity = state_dict['angular_velocity']
        if 'angular_velocity' in state_dict or 'linear_velocity' in state_dict:
            self._pybullet_client.resetBaseVelocity(self.__block_id,
                                                    linear_velocity,
                                                    angular_velocity)
        return

    def do_intervention(self, variable_name, variable_value):
        #TODO: discuss handling collisions with fingers with Fred
        if variable_name == 'position':
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, variable_value, orientation
            )
        elif variable_name == 'orientation':
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self.__block_id, position, variable_value
            )
        elif variable_name == 'mass':
            self._pybullet_client.changeDynamics(self.__block_id, -1, mass=variable_value)
            self.__mass = variable_value
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.__block_id, -1,
                                                    rgbaColor=np.append(variable_value, 1))
            self.__color = variable_value
        elif variable_name == 'linear_velocity':
            _, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            self._pybullet_client.resetBaseVelocity(self.__block_id,
                                                    variable_value,
                                                    angular_velocity)
        elif variable_name == 'angular_velocity':
            linear_velocity, _ = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            self._pybullet_client.resetBaseVelocity(self.__block_id,
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
                self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            state["type"] = self.__type_id
            state["position"] = np.array(position)
            state["orientation"] = np.array(orientation)
            linear_velocity, angular_velocity = \
                self._pybullet_client.getBaseVelocity(self.__block_id)
            state["linear_velocity"] = np.array(linear_velocity)
            state["angular_velocity"] = np.array(angular_velocity)
            state["mass"] = self.__mass
            state["size"] = self.__size
            state["color"] = self.__color
        elif state_type == 'list':
            state = []
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
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
            position, orientation = self._pybullet_client.getBasePositionAndOrientation(
                self.__block_id
            )
            return orientation
        elif variable_name == 'linear_velocity':
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
            return linear_velocity

        elif variable_name == 'angular_velocity':
            linear_velocity, angular_velocity = self._pybullet_client.getBaseVelocity(
                self.__block_id)
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
        self._pybullet_client.resetBasePositionAndOrientation(
            self.__block_id, position, orientation
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
        pybullet_client,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        color=np.array([0, 0, 0])
    ):
        """

        :param pybullet_client:
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
        self.color = color
        self.shape_id = pybullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, halfExtents=np.array(size)/2)
        self.block_id = pybullet_client.createMultiBody(
            baseCollisionShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=0
        )
        super(StaticCuboid, self).__init__(pybullet_client, name, self.block_id)
        self._pybullet_client.changeVisualShape(self.block_id, -1,
                                                rgbaColor=np.append(self.color, 1))
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self._name + "_type"] = np.array([0])
        self.lower_bounds[self._name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self._name + "_orientation"] = \
            np.array([-10] * 4)
        self.lower_bounds[self._name + "_size"] = \
            np.array([0.065, 0.065, 0.065])
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
            self._pybullet_client.removeBody(self.block_id)
            self.shape_id = self._pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(state_dict['size']) / 2)
            self.block_id = self._pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation,
                baseMass=0
            )
            self.size = state_dict['size']
        elif 'position' in state_dict or 'orientation' in state_dict:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, self.orientation
            )
        if 'color' in state_dict:
            self.color = state_dict['color']
            self._pybullet_client.changeVisualShape(self.block_id, -1,
                                                    rgbaColor=np.append(state_dict['color'], 1))
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
            self._pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, self.orientation
            )
        elif variable_name == 'orientation':
            self.orientation = variable_value
            self._pybullet_client.resetBasePositionAndOrientation(
                self.block_id, self.position, self.orientation
            )
        elif variable_name == 'size':
            self._pybullet_client.removeBody(self.block_id)
            self.shape_id = self._pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX, halfExtents=np.array(variable_value) / 2)
            self.block_id = self._pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.shape_id,
                basePosition=self.position,
                baseOrientation=self.orientation,
                baseMass=0
            )
            self._pybullet_client.changeVisualShape(self.block_id, -1,
                                                    rgbaColor=np.append(self.color, 1))
            self.size = variable_value
        elif variable_name == 'color':
            self._pybullet_client.changeVisualShape(self.block_id, -1,
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
        self._pybullet_client.resetBasePositionAndOrientation(
            self.block_id, position, orientation
        )
        self.position = position
        self.orientation = orientation
        return







