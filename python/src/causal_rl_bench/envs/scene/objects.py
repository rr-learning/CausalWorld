import pybullet
import numpy as np


class RigidObject(object):
    def __init__(self, pybullet_client, name):
        self.pybullet_client = pybullet_client
        self.name = name

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

    def is_not_fixed(self):
        raise NotImplementedError()

    def get_variable_state(self, variable_name):
        raise NotImplementedError()


class Cuboid(RigidObject):
    def __init__(
        self,
        pybullet_client,
        name, size=np.array([0.065, 0.065, 0.065]),
        position=np.array([0.0, 0.0, 0.0425]),
        orientation=np.array([0, 0, 0, 1]),
        mass=0.08, colour=np.array([1, 0, 0])
    ):
        """
        Import the block

        Args:
            position (list): where in xyz space should the block
                be imported
            orientation (list): initial orientation quaternion of the block
            half_size (float): how large should this block be
            mass (float): how heavy should this block be
        """
        #TODO: intervene on friction as well
        super(Cuboid, self).__init__(pybullet_client, name)
        self.type_id = 0
        self.mass = mass
        self.size = size
        self.not_fixed = True
        self.colour = colour
        self.block_id = self.pybullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_BOX, halfExtents=np.array(size)/2)
        self.block = self.pybullet_client.createMultiBody(
            baseCollisionShapeIndex=self.block_id,
            basePosition=position,
            baseOrientation=orientation,
            baseMass=mass
        )
        self.pybullet_client.changeVisualShape(self.block, -1, rgbaColor=np.append(self.colour, 1))
        #specifying bounds
        self.lower_bounds = dict()
        self.upper_bounds = dict()
        self.lower_bounds[self.name + "_type"] = np.array([0])
        self.lower_bounds[self.name + "_position"] = \
            np.array([-0.5, -0.5, 0])
        self.lower_bounds[self.name + "_orientation"] = \
            np.array([-10] * 4)
        self.lower_bounds[self.name + "_linear_velocity"] = \
            np.array([-0.5] * 3)
        self.lower_bounds[self.name + "_angular_velocity"] = \
            np.array([-0.5] * 3)
        self.lower_bounds[self.name + "_mass"] = \
            np.array([0])
        self.lower_bounds[self.name + "_size"] = \
            np.array([0.065, 0.065, 0.065])
        self.lower_bounds[self.name + "_colour"] = \
            np.array([0]*3)

        self.upper_bounds[self.name + "_type"] = np.array([10])
        self.upper_bounds[self.name + "_position"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_orientation"] = \
            np.array([10] * 4)
        self.upper_bounds[self.name + "_linear_velocity"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_angular_velocity"] = \
            np.array([0.5] * 3)
        self.upper_bounds[self.name + "_mass"] = \
            np.array([0.2])
        self.upper_bounds[self.name + "_size"] = \
            np.array([0.1, 0.1, 0.1])
        self.upper_bounds[self.name + "_colour"] = \
            np.array([1]*3)
        self._state_variable_names = ['type', 'position',
                                      'orientation', 'linear_velocity',
                                      'angular_velocity', 'mass',
                                      'size', 'colour']
        self._simplified_state_variable_names = ['position',
                                                 'orientation',
                                                 'linear_velocity',
                                                 'angular_velocity']
        self._state_variable_sizes = []
        self.state_size = 0
        for state_variable_name in self._state_variable_names:
            self._state_variable_sizes.append(
                self.upper_bounds[self.name + "_" +
                                  state_variable_name].shape[0])
            self.state_size += self._state_variable_sizes[-1]

    def set_full_state(self, new_state):
        #form dict first
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
            position, orientation = self.pybullet_client.\
                getBasePositionAndOrientation(
                self.block
            )
        if 'position' in state_dict:
            position = state_dict['position']
        if 'orientation' in state_dict:
            orientation = state_dict['orientation']
        if 'mass' in state_dict:
            self.mass = state_dict['mass']
        if 'size' in state_dict:
            self.pybullet_client.removeBody(self.block)
            self.block_id = self.pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=np.array(state_dict['size']) / 2)
            self.block = self.pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=self.mass
            )
            self.size = state_dict['size']
        elif 'position' in state_dict or 'orientation' in state_dict:
            self.pybullet_client.resetBasePositionAndOrientation(
                self.block, position, orientation
            )
        elif 'mass' in state_dict:
            self.pybullet_client.changeDynamics(self.block, -1, mass=self.mass)

        if 'colour' in  state_dict:
            self.colour = state_dict['colour']
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor=np.append(state_dict['colour'], 1))
        if ('linear_velocity' in state_dict) ^ \
                ('angular_velocity' in state_dict):
            linear_velocity, angular_velocity = self.pybullet_client.getBaseVelocity(
                self.block)
        if 'linear_velocity' in state_dict:
            linear_velocity = state_dict['linear_velocity']
        if 'angular_velocity' in state_dict:
            angular_velocity = state_dict['angular_velocity']
        if 'angular_velocity' in state_dict or 'linear_velocity' in state_dict:
            self.pybullet_client.resetBaseVelocity(self.block,
                                                   linear_velocity,
                                                   angular_velocity)
        return

    def do_intervention(self, variable_name, variable_value):
        #TODO: discuss handling collisions with fingers with Fred
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
        elif variable_name == 'mass':
            self.pybullet_client.changeDynamics(self.block, -1, mass=variable_value)
            self.mass = variable_value
        elif variable_name == 'size':
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            self.pybullet_client.removeBody(self.block)
            self.block_id = self.pybullet_client.createCollisionShape(
                shapeType=pybullet.GEOM_BOX, halfExtents=np.array(variable_value) / 2)
            self.block = self.pybullet_client.createMultiBody(
                baseCollisionShapeIndex=self.block_id,
                basePosition=position,
                baseOrientation=orientation,
                baseMass=self.mass
            )
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor=np.append(self.colour, 1))
            self.size = variable_value
        elif variable_name == 'colour':
            self.pybullet_client.changeVisualShape(self.block, -1,
                                                   rgbaColor=np.append(variable_value, 1))
            self.colour = variable_value
        elif variable_name == 'linear_velocity':
            _, angular_velocity = self.pybullet_client.getBaseVelocity(
                self.block)
            self.pybullet_client.resetBaseVelocity(self.block,
                                                   variable_value,
                                                   angular_velocity)
        elif variable_name == 'angular_velocity':
            linear_velocity, _ = self.pybullet_client.getBaseVelocity(
                self.block)
            self.pybullet_client.resetBaseVelocity(self.block,
                                                   linear_velocity,
                                                   variable_value)
        #TODO: implement intervention on shape id itself
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
            linear_velocity, angular_velocity = self.pybullet_client.getBaseVelocity(self.block)
            state[self.name + "_linear_velocity"] = np.array(linear_velocity)
            state[self.name + "_angular_velocity"] = np.array(angular_velocity)
            state[self.name + "_mass"] = self.mass
            state[self.name + "_size"] = self.size
            state[self.name + "_colour"] = self.colour
        elif state_type == 'list':
            state = []
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            linear_velocity, angular_velocity = self.pybullet_client.getBaseVelocity(
                self.block)
            for name in self._state_variable_names:
                if name == 'type':
                    state.append(self.type_id)
                elif name == 'position':
                    state.extend(position)
                elif name == 'orientation':
                    state.extend(orientation)
                elif name == 'linear_velocity':
                    state.extend(linear_velocity)
                elif name == 'angular_velocity':
                    state.extend(angular_velocity)
                elif name == 'mass':
                    state.append(self.mass)
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
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            return position

        elif variable_name == 'orientation':
            position, orientation = self.pybullet_client.getBasePositionAndOrientation(
                self.block
            )
            return orientation
        elif variable_name == 'linear_velocity':
            linear_velocity, angular_velocity = self.pybullet_client.getBaseVelocity(
                self.block)
            return linear_velocity

        elif variable_name == 'angular_velocity':
            linear_velocity, angular_velocity = self.pybullet_client.getBaseVelocity(
                self.block)
            return angular_velocity

        elif variable_name == 'mass':
            return self.mass
        elif variable_name == 'size':
            return self.size
        elif variable_name == 'colour':
            return self.colour
        else:
            raise Exception("variable name is not supported")

    def get_state_variable_names(self):
        return self._state_variable_names

    def is_not_fixed(self):
        return self.not_fixed

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_state_size(self):
        return self.state_size

    def set_pose(self, position, orientation):
        self.pybullet_client.resetBasePositionAndOrientation(
            self.block, position, orientation
        )
        return








