from causal_world.envs.scene.observations import StageObservations
from causal_world.envs.scene.objects import Cuboid, StaticCuboid, MeshObject
from causal_world.envs.scene.silhouette import SCuboid, SSphere, SMeshObject
from causal_world.utils.state_utils import get_intersection, get_bounding_box_volume
import math
import numpy as np
import pybullet
from causal_world.configs.world_constants import WorldConstants
from collections import OrderedDict


class Stage(object):

    def __init__(self, observation_mode, normalize_observations,
                 pybullet_client_full_id, pybullet_client_w_goal_id,
                 pybullet_client_w_o_goal_id, cameras, camera_indicies):
        """
        This class represents the stage object, where it handles all the arena
        functionalities including the objects and silhouettes existing in
        the arena.

        :param observation_mode: (str) should be "structured" or "pixel"
        :param normalize_observations: (bool) to normalize the observations or
                                              not.
        :param pybullet_client_full_id: (int) pybullet client if visualization
                                              is enabled.
        :param pybullet_client_w_goal_id: (int) pybullet client with the goal
                                                in the image without tool
                                                objects.
        :param pybullet_client_w_o_goal_id: (int) pybullet client without the
                                                  goal, only tool blocks.
        :param cameras: (list) list of causal_world.robot.Camera object
                               specifying the cameras mounted on top of the
                               trifinger robot.
        :param camera_indicies: (list) list of integers of the order of cameras
                                       to be used.
        """
        self._rigid_objects = OrderedDict()
        self._visual_objects = OrderedDict()
        self._observation_mode = observation_mode
        self._pybullet_client_full_id = pybullet_client_full_id
        self._pybullet_client_w_goal_id = pybullet_client_w_goal_id
        self._pybullet_client_w_o_goal_id = pybullet_client_w_o_goal_id
        self._camera_indicies = camera_indicies
        self._normalize_observations = normalize_observations
        self._stage_observations = None
        self._name_keys = []
        self._default_gravity = [0, 0, -9.81]
        self._current_gravity = np.array(self._default_gravity)
        self._visual_object_client_instances = []
        self._rigid_objects_client_instances = []
        if self._pybullet_client_full_id is not None:
            self._visual_object_client_instances.append(
                self._pybullet_client_full_id)
            self._rigid_objects_client_instances.append(
                self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            self._rigid_objects_client_instances.append(
                self._pybullet_client_w_o_goal_id)
        if self._pybullet_client_w_goal_id is not None:
            self._visual_object_client_instances.append(
                self._pybullet_client_w_goal_id)
        self._cameras = cameras
        self._goal_image = None
        return

    def get_floor_height(self):
        """

        :return: (float) returns the floor height.
        """
        return WorldConstants.FLOOR_HEIGHT

    def get_arena_bb(self):
        """

        :return: (list) list of the lower bound (x, y, z) and the upper bound
                        (x, y, z) so (2, 3) shape.
        """
        return  WorldConstants.ARENA_BB

    def get_rigid_objects(self):
        """

        :return: (dict) returns the rigid objects in the arena currently.
        """
        return self._rigid_objects

    def get_visual_objects(self):
        """

        :return: (dict) returns the visual objects in the arena currently.
        """
        return self._visual_objects

    def get_full_env_state(self):
        """

        :return: (dict) returns a dict specifying everything about the current
                        state of the arena.
        """
        env_state = {}
        env_state['rigid_objects'] = []
        for rigid_object_key in self._rigid_objects:
            if isinstance(self._rigid_objects[rigid_object_key], Cuboid):
                env_state['rigid_objects'].append([
                    'cube',
                    self._rigid_objects[rigid_object_key].get_recreation_params(
                    )
                ])
            if isinstance(self._rigid_objects[rigid_object_key], StaticCuboid):
                env_state['rigid_objects'].append([
                    'static_cube',
                    self._rigid_objects[rigid_object_key].get_recreation_params(
                    )
                ])
            if isinstance(self._rigid_objects[rigid_object_key], MeshObject):
                env_state['rigid_objects'].append([
                    'mesh',
                    self._rigid_objects[rigid_object_key].get_recreation_params(
                    )
                ])
        env_state['visual_objects'] = []
        for visual_object_key in self._visual_objects:
            if isinstance(self._visual_objects[visual_object_key], SCuboid):
                env_state['visual_objects'].append([
                    'cube', self._visual_objects[visual_object_key].
                    get_recreation_params()
                ])
            if isinstance(self._visual_objects[visual_object_key], SSphere):
                env_state['visual_objects'].append([
                    'sphere', self._visual_objects[visual_object_key].
                    get_recreation_params()
                ])
            if isinstance(self._visual_objects[visual_object_key], SMeshObject):
                env_state['visual_objects'].append([
                    'mesh', self._visual_objects[visual_object_key].
                    get_recreation_params()
                ])
        env_state['arena_variable_values'] = \
            self.get_current_variable_values_for_arena()
        return env_state

    def set_full_env_state(self, env_state):
        """

        :param env_state: (dict) dict specifying everything about the current
                                 state of the arena, usually obtained through
                                 get_full_env_state function.

        :return:
        """
        self.remove_everything()
        for rigid_object_info in env_state['rigid_objects']:
            if rigid_object_info[0] == 'mesh':
                self.add_rigid_mesh_object(**rigid_object_info[1])
            else:
                self.add_rigid_general_object(shape=rigid_object_info[0],
                                              **rigid_object_info[1])
        for visual_object_info in env_state['visual_objects']:
            if visual_object_info[0] == 'mesh':
                self.add_silhoutte_mesh_object(**visual_object_info[1])
            else:
                self.add_silhoutte_general_object(shape=visual_object_info[0],
                                                  **visual_object_info[1])
        self.apply_interventions(env_state['arena_variable_values'])
        self._stage_observations.rigid_objects = self._rigid_objects
        self._stage_observations.visual_objects = self._visual_objects
        return

    def add_rigid_general_object(self, name, shape, **object_params):
        """

        :param name: (str) a str specifying a unique name of the rigid object.
        :param shape: (str) specifying "cube" or "static_cube" for now.
        :param object_params: (params) depends on the parameters used for
                                       constructing the corresponding object.

        :return:
        """
        if name in self._name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self._name_keys.append(name)
        if shape == "cube":
            self._rigid_objects[name] = Cuboid(
                self._rigid_objects_client_instances, name, **object_params)
        elif shape == "static_cube":
            self._rigid_objects[name] = StaticCuboid(
                self._rigid_objects_client_instances, name, **object_params)
        else:
            raise Exception("shape is not yet implemented")
        return

    def remove_general_object(self, name):
        """

        :param name: (str) a str specifying a unique name of the object
                           to remove from the arena.

        :return:
        """
        if name not in self._name_keys:
            raise Exception("name does not exists as key for scene objects")
        else:
            self._name_keys.remove(name)
        if name in self._rigid_objects.keys():
            self._rigid_objects[name].remove()

            del self._rigid_objects[name]
        elif name in self._visual_objects.keys():
            self._visual_objects[name].remove()
            del self._visual_objects[name]
        return

    def remove_everything(self):
        """
        removes all the objects and visuals from the arena.

        :return:
        """
        current_objects = list(self._rigid_objects.keys()) + \
                          list(self._visual_objects.keys())
        current_objects = current_objects[::-1]
        for name in current_objects:
            self.remove_general_object(name)
        return

    def add_rigid_mesh_object(self, name, filename, **object_params):
        """

        :param name: (str) a str specifying a unique name of the mesh object.
        :param filename: (str) a str specifying the location of the .obj file.
        :param object_params: (params) depends on the parameters used for
                                       constructing the corresponding object.

        :return:
        """
        if name in self._name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self._name_keys.append(name)
        self._rigid_objects[name] = MeshObject(
            self._rigid_objects_client_instances, name, filename,
            **object_params)
        return

    def add_silhoutte_general_object(self, name, shape, **object_params):
        """

        :param name: (str) specifying a unique name of the visual object.
        :param shape: (str) specifying "cube" or "sphere" for now.
        :param object_params: (params) depends on the parameters used for
                                       constructing the corresponding object.

        :return:
        """
        if name in self._name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self._name_keys.append(name)
        if shape == "cube":
            self._visual_objects[name] = SCuboid(
                self._visual_object_client_instances, name, **object_params)
        elif shape == "sphere":
            self._visual_objects[name] = SSphere(
                self._visual_object_client_instances, name, **object_params)
        else:
            raise Exception("shape is not implemented yet")
        return

    def add_silhoutte_mesh_object(self, name, filename, **object_params):
        """

        :param name: (str) specifying a unique name of the mesh visual object.
        :param filename: (str) a str specifying the location of the .obj file.
        :param object_params: (params) depends on the parameters used for
                                       constructing the corresponding object.

        :return:
        """
        if name in self._name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self._name_keys.append(name)
        self._visual_objects[name] = SMeshObject(
            self._visual_object_client_instances, name, filename,
            **object_params)
        return

    def finalize_stage(self):
        """
        finalizes the observation space of the environment after adding all
        the objects and visuals in the stage.

        :return:
        """
        if self._observation_mode == "pixel":
            self._stage_observations = StageObservations(
                self._rigid_objects,
                self._visual_objects,
                self._observation_mode,
                self._normalize_observations,
                cameras=self._cameras,
                camera_indicies=self._camera_indicies)
            self.update_goal_image()
        else:
            self._stage_observations = StageObservations(
                self._rigid_objects, self._visual_objects,
                self._observation_mode, self._normalize_observations)
        return

    def select_observations(self, observation_keys):
        """
        selects the observations to be returned by the environment.

        :param observation_keys: (list) list of str that specifies the
                                        observation keys to be returned and
                                        calculated by the environment.

        :return:
        """
        self._stage_observations.reset_observation_keys()
        self._stage_observations.initialize_observations()
        for key in observation_keys:
            self._stage_observations.add_observation(key)
        self._stage_observations.set_observation_spaces()

    def get_full_state(self, state_type='list'):
        """

        :param state_type: (str) 'list' or 'dict' specifying to return the
                                  state as a dict with the state name as a key
                                  or just a concatenated list.

        :return: (list or dict) depending on the arg state_type, returns
                                the full state of the stage itself.
        """
        if state_type == 'list':
            stage_state = []
        elif state_type == 'dict':
            stage_state = dict()
        else:
            raise Exception("type is not supported")
        for name in self._name_keys:
            if name in self._rigid_objects:
                object = self._rigid_objects[name]
            elif name in self._visual_objects:
                object = self._visual_objects[name]
            else:
                raise Exception("possible error here")
            if state_type == 'list':
                stage_state.extend(object.get_state(state_type='list'))
            elif state_type == 'dict':
                stage_state[name] = object.get_state(state_type='dict')
        return stage_state

    def set_full_state(self, new_state):
        """

        :param new_state: (dict) specifies the full state of all the objects
                                 in the arena.

        :return:
        """
        #TODO: under the assumption that the new state has the same number of objects
        start = 0
        for name in self._name_keys:
            if name in self._rigid_objects:
                object = self._rigid_objects[name]
                end = start + object.get_state_size()
                object.set_full_state(new_state[start:end])
            elif name in self._visual_objects:
                object = self._visual_objects[name]
                end = start + object.get_state_size()
                object.set_full_state(new_state[start:end])
            start = end
        if self._observation_mode == "pixel":
            self.update_goal_image()
        return

    def set_objects_pose(self, names, positions, orientations):
        """

        :param names: (list) list of object names to set their positions and
                             orientations.
        :param positions: (list) corresponding list of positions of
                                 objects to be set.
        :param orientations: (list) corresponding list of orientations of
                                    objects to be set.

        :return:
        """
        for i in range(len(names)):
            name = names[i]
            if name in self._rigid_objects:
                object = self._rigid_objects[name]
                object.set_pose(positions[i], orientations[i])
            elif name in self._visual_objects:
                object = self._visual_objects[name]
                object.set_pose(positions[i], orientations[i])
            else:
                raise Exception("Object {} doesnt exist".format(name))
        if self._observation_mode == "pixel":
            self.update_goal_image()
        return

    def get_current_observations(self, helper_keys):
        """

        :param helper_keys: (list) list of observation keys that are not
                                   part of the observation space but still
                                   needed to be returned to calculate further
                                   observations or for a dense reward function
                                   calculation.

        :return: (dict) returns the current observations where the keys
                        corresponds to the observation key.
        """
        return self._stage_observations.get_current_observations(helper_keys)

    def get_observation_spaces(self):
        """

        :return: (gym.spaces.Box) returns the current observation space of the
                                  environment.
        """
        return self._stage_observations.get_observation_spaces()

    def random_position(self,
                        height_limits=(0.05, 0.15),
                        angle_limits=(-2 * math.pi, 2 * math.pi),
                        radius_limits=(0.0, 0.15),
                        allowed_section=np.array([[-0.5, -0.5, 0],
                                                  [0.5, 0.5, 0.5]])):
        """

        :param height_limits: (tuple) tuple of two values for low bound
                                      and upper bound.
        :param angle_limits: (tuple) tuple of two values for low bound
                                      and upper bound, theta in
                                      polar coordinates.
        :param radius_limits: (tuple) tuple of two values for low bound
                                      and upper bound, radius in
                                      polar coordinates.
        :param allowed_section: (nd.array) array of two sublists for low bound
                                           and upper bound (x, y, z) of
                                           restricted area for sampling, the
                                           shape of the input is basically (2,2).

        :return: (list) returns a cartesian random position in the arena
                        (x, y, z).
        """
        satisfying_constraints = False
        while not satisfying_constraints:
            angle = np.random.uniform(*angle_limits)
            # for uniform sampling with respect to the disc area use scaling
            radial_distance = np.sqrt(
                np.random.uniform(radius_limits[0]**2, radius_limits[1]**2))

            if isinstance(height_limits, (int, float)):
                height_z = height_limits
            else:
                height_z = np.random.uniform(*height_limits)

            object_position = [
                radial_distance * math.cos(angle),
                radial_distance * math.sin(angle),
                height_z,
            ]
            #check if satisfying_constraints
            if np.all(object_position > allowed_section[0]) and \
                    np.all(object_position < allowed_section[1]):
                satisfying_constraints = True

        return object_position

    def get_current_object_keys(self):
        """

        :return: (list) returns the names of the rigid objects and visual
                        objects concatenated.
        """
        return list(self._rigid_objects.keys()) + \
               list(self._visual_objects.keys())

    def object_intervention(self, key, interventions_dict):
        """

        :param key: (str) the unique name of the rigid or visual
                          object to intervene on.
        :param interventions_dict: (dict) dict specifying the intervention to
                                          be performed.

        :return:
        """
        if key in self._rigid_objects:
            object = self._rigid_objects[key]
        elif key in self._visual_objects:
            object = self._visual_objects[key]
        else:
            raise Exception(
                "The key {} passed doesn't exist in the stage yet".format(key))
        object.apply_interventions(interventions_dict)
        if self._observation_mode == "pixel":
            self.update_goal_image()
        return

    def get_current_variable_values_for_arena(self):
        """

        :return: (dict) returns all the exposed variables and their values
                        in the environment's stage except for objects.
        """
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        variable_params = dict()
        variable_params["floor_color"] = \
            pybullet.getVisualShapeData(WorldConstants.FLOOR_ID,
                                        physicsClientId=client)[0][7][:3]
        variable_params["floor_friction"] = \
            pybullet.getDynamicsInfo(WorldConstants.FLOOR_ID, -1,
                                     physicsClientId=client)[1]

        variable_params["stage_color"] = \
            pybullet.getVisualShapeData(WorldConstants.STAGE_ID,
                                        physicsClientId=client)[0][7][:3]
        variable_params["stage_friction"] = \
            pybullet.getDynamicsInfo(WorldConstants.STAGE_ID, -1,
                                     physicsClientId=client)[1]

        variable_params["gravity"] = \
            self._current_gravity
        return variable_params

    def get_current_variable_values_for_objects(self):
        """

        :return: (dict) returns all the exposed variables and their values
                        in the environment's stage for objects only.
        """
        return self.get_full_state(state_type='dict')

    def get_current_variable_values(self):
        """

        :return: (dict) returns all the exposed variables and their values
                        in the environment's stage.
        """
        variable_params = self.get_current_variable_values_for_arena()
        variable_params.update(self.get_current_variable_values_for_objects())
        return variable_params

    def apply_interventions(self, interventions_dict):
        """

        :param interventions_dict: (dict) dict specifying the intervention to
                                          be performed.

        :return:
        """
        for intervention in interventions_dict:
            if isinstance(interventions_dict[intervention], dict):
                self.object_intervention(intervention,
                                         interventions_dict[intervention])
            elif intervention == "floor_color":
                for client in self._visual_object_client_instances:
                    pybullet.changeVisualShape(
                        WorldConstants.FLOOR_ID,
                        -1,
                        rgbaColor=np.append(interventions_dict[intervention],
                                            1),
                        physicsClientId=client)
                for client in self._rigid_objects_client_instances:
                    pybullet.changeVisualShape(
                        WorldConstants.FLOOR_ID,
                        -1,
                        rgbaColor=np.append(interventions_dict[intervention],
                                            1),
                        physicsClientId=client)
            elif intervention == "stage_color":
                for client in self._visual_object_client_instances:
                    pybullet.changeVisualShape(
                        WorldConstants.STAGE_ID,
                        -1,
                        rgbaColor=np.append(interventions_dict[intervention],
                                            1),
                        physicsClientId=client)
                for client in self._rigid_objects_client_instances:
                    pybullet.changeVisualShape(
                        WorldConstants.STAGE_ID,
                        -1,
                        rgbaColor=np.append(interventions_dict[intervention],
                                            1),
                        physicsClientId=client)
            elif intervention == "stage_friction":
                for client in self._rigid_objects_client_instances:
                    pybullet.changeDynamics(
                        bodyUniqueId=WorldConstants.STAGE_ID,
                        linkIndex=-1,
                        lateralFriction=interventions_dict[intervention],
                        physicsClientId=client)
            elif intervention == "floor_friction":
                for client in self._rigid_objects_client_instances:
                    pybullet.changeDynamics(
                        bodyUniqueId=WorldConstants.FLOOR_ID,
                        linkIndex=-1,
                        lateralFriction=interventions_dict[intervention],
                        physicsClientId=client)
            elif intervention == "gravity":
                for client in self._rigid_objects_client_instances:
                    pybullet.setGravity(interventions_dict[intervention][0],
                                        interventions_dict[intervention][1],
                                        interventions_dict[intervention][2],
                                        physicsClientId=client)
                self._current_gravity = interventions_dict[intervention]
            else:
                raise Exception("The intervention on stage "
                                "is not supported yet")
        if self._observation_mode == "pixel":
            self.update_goal_image()
        return

    def get_object_full_state(self, key):
        """

        :param key: (str) specifying the name of the object to return
                          its state.

        :return: (dict) specifies the state of the object queried.
        """
        if key in self._rigid_objects:
            return self._rigid_objects[key].get_state('dict')
        elif key in self._visual_objects:
            return self._visual_objects[key].get_state('dict')
        else:
            raise Exception(
                "The key {} passed doesn't exist in the stage yet".format(key))

    def get_object_state(self, key, state_variable):
        """

        :param key: (str) specifying the name of the object to return
                          its state's variable value.
        :param state_variable: (str) specifying the variable's name of the
                                     object.

        :return: (nd.array) returns the variable's value of the object queried.
        """
        if key in self._rigid_objects:
            return np.array(
                self._rigid_objects[key].get_variable_state(state_variable))
        elif key in self._visual_objects:
            return np.array(
                self._visual_objects[key].get_variable_state(state_variable))
        else:
            raise Exception(
                "The key {} passed doesn't exist in the stage yet".format(key))

    def get_object(self, key):
        """

        :param key: (str) specifying the name of the object to return it.

        :return: (causal_world.RigidObject or causal_world.SilhouetteObject)
                 object to return.
        """
        if key in self._rigid_objects:
            return self._rigid_objects[key]
        elif key in self._visual_objects:
            return self._visual_objects[key]
        else:
            raise Exception(
                "The key {} passed doesn't exist in the stage yet".format(key))

    def are_blocks_colliding(self, block1, block2):
        """

        :param block1: (causal_world.RigidObject) first block.
        :param block2: (causal_world.RigidObject) second block.

        :return: (bool) true if the two blocks passed are colliding.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if (contact[1] == block1._block_ids[0] and
                contact[2] == block2._block_ids[0]) or \
                    (contact[2] == block1._block_ids[0] and
                     contact[1] == block2._block_ids[0]):
                return True
        return False

    def check_stage_free_of_colliding_blocks(self):
        """

        :return: (bool) true if the stage is free of collisions of blocks.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if contact[1] > 3 and contact[2] > 3:
                return False
        return True

    def is_colliding_with_stage(self, block1):
        """

        :param block1: (causal_world.RigidObject) first block.

        :return: (bool) true if the stage is free of collisions between blocks.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if (contact[1] == block1._block_ids[0] and contact[2] ==
                WorldConstants.STAGE_ID) or \
                    (contact[2] == block1._block_ids[0] and contact[1] ==
                     WorldConstants.STAGE_ID):
                return True
        return False

    def is_colliding_with_floor(self, block1):
        """

        :param block1: (causal_world.RigidObject) first block.

        :return: (bool) true if the block is colliding with the floor.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if (contact[1] == block1._block_ids[0] and contact[2] ==
                WorldConstants.FLOOR_ID) or \
                    (contact[2] == block1._block_ids[0] and contact[1] ==
                     WorldConstants.FLOOR_ID):
                return True
        return False

    def get_normal_interaction_force_between_blocks(self, block1, block2):
        """

        :param block1: (causal_world.RigidObject) first block.
        :param block2: (causal_world.RigidObject) second block.

        :return: (float) normal interaction force between blocks or None if
                         no interaction.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if (contact[1] == block1._block_ids[0] and contact[2] ==
                block2._block_ids[0]) or \
                    (contact[2] == block1._block_ids[0] and contact[1] ==
                     block2._block_ids[0]):
                return contact[9] * np.array(contact[7])
        return None

    def add_observation(self,
                        observation_key,
                        lower_bound=None,
                        upper_bound=None):
        """

        :param observation_key: (str) new observation key to be added.
        :param lower_bound: (nd.array) low bound of the observation.
        :param upper_bound: (nd.array) upper bound of the observation.

        :return:
        """
        self._stage_observations.add_observation(observation_key, lower_bound,
                                                 upper_bound)
        return

    def normalize_observation_for_key(self, observation, key):
        """

        :param observation: (nd.array) observation to normalize.
        :param key: (str) observation key to be normalized.

        :return: (nd.array) normalized observation.
        """
        return self._stage_observations.normalize_observation_for_key(observation,
                                                                      key)

    def denormalize_observation_for_key(self, observation, key):
        """

        :param observation: (nd.array) observation to denormalize.
        :param key: (str) observation key to be denormalized.

        :return: (nd.array) denormalized observation.
        """
        return self._stage_observations.denormalize_observation_for_key(observation,
                                                                        key)

    def get_current_goal_image(self):
        """

        :return: (nd.array) returns the goal images concatenated if 'pixel' mode
                            is enabled.
        """
        return self._goal_image

    def update_goal_image(self):
        """
        updated the goal image.

        :return:
        """
        self._goal_image = self._stage_observations.get_current_goal_image()
        return

    def check_feasiblity_of_stage(self):
        """
        This function checks the feasibility of the current state of the stage
        (i.e checks if any of the bodies in the simulation are in a penetration
        mode)

        :return: (bool) A boolean indicating whether the stage is in a
                        collision state or not. As well as if the visual
                        objects are outside of the bounding box or not.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self._rigid_objects_client_instances[0]):
            if contact[8] < -0.03:
                return False
        #check if all the visual objects are within the bb og the available arena
        for visual_object in self._visual_objects:
            if get_intersection(self._visual_objects[visual_object].
                                   get_bounding_box(),
                                self.get_stage_bb())/\
                    self._visual_objects[visual_object].get_volume() < 0.50:
                return False
            if self._visual_objects[visual_object].get_bounding_box()[0][-1] < -0.01:
                return False
        return True

    def get_stage_bb(self):
        """

        :return: (tuple) first element indicates the lower bound the stage arena
                         (x,y, z) and second element indicated the upper bound
                         similarly.
        """
        return (tuple(WorldConstants.ARENA_BB[0]),
                tuple(WorldConstants.ARENA_BB[1]))
