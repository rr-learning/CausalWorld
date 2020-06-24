from causal_rl_bench.envs.scene.observations import StageObservations
from causal_rl_bench.envs.scene.objects import Cuboid, StaticCuboid, MeshObject
from causal_rl_bench.envs.scene.silhouette import SCuboid, SSphere, SMeshObject
from causal_rl_bench.utils.state_utils import get_intersection
import math
import numpy as np
import pybullet


class Stage(object):
    def __init__(self, observation_mode,
                 normalize_observations,
                 pybullet_client_full_id,
                 pybullet_client_w_goal_id,
                 pybullet_client_w_o_goal_id):
        """

        :param observation_mode:
        :param normalize_observations:
        :param pybullet_client_full:
        :param pybullet_client_w_goal:
        :param pybullet_client_w_o_goal:
        """
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.observation_mode = observation_mode
        self._pybullet_client_full_id = pybullet_client_full_id
        self._pybullet_client_w_goal_id = pybullet_client_w_goal_id
        self._pybullet_client_w_o_goal_id = pybullet_client_w_o_goal_id
        #TODO: move the ids from here
        self.floor_id = 2
        self.stage_id = 3
        self.normalize_observations = normalize_observations
        self.stage_observations = None
        #TODO: to be deleted below
        # self.latest_full_state = None
        # self.latest_observations = None
        self.floor_height = 0.01
        #TODO: discuss this with Manuel and Felix for the bounds
        self.floor_inner_bounding_box = np.array(
            [[-0.15, -0.15, self.floor_height], [0.15, 0.15, 0.3]])
        self.name_keys = []
        self.default_gravity = [0, 0, -9.81]
        self.current_gravity = np.array(self.default_gravity)
        self.visual_object_client_instances = []
        self.rigid_objects_client_instances = []
        if self._pybullet_client_full_id is not None:
            self.visual_object_client_instances.append(
                self._pybullet_client_full_id)
            self.rigid_objects_client_instances.append(
                self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            self.rigid_objects_client_instances.append(
                self._pybullet_client_w_o_goal_id)
        if self._pybullet_client_w_goal_id is not None:
            self.visual_object_client_instances.append(
                self._pybullet_client_w_goal_id)
        self.goal_image = None
        return

    def get_full_memory(self):
        memory = {}
        memory['rigid_objects'] = []
        for rigid_object_key in self.rigid_objects:
            memory['rigid_objects'].append(['cube',
                                            self.rigid_objects
                                            [rigid_object_key].
                                           get_recreation_params()])
        memory['visual_objects'] = []
        for visual_object_key in self.visual_objects:
            memory['visual_objects'].append(['cube',
                                            self.visual_objects
                                            [visual_object_key].
                                           get_recreation_params()])
        # memory['current_gravity'] = copy.deepcopy(self.current_gravity)
        # self.latest_full_state = None
        # self.latest_observations = None
        return memory

    def set_full_memory(self, memory):
        self.remove_everything()
        for rigid_object_info in memory['rigid_objects']:
            self.add_rigid_general_object(shape=rigid_object_info[0],
                                          **rigid_object_info[1])
        for visual_object_info in memory['visual_objects']:
            self.add_silhoutte_general_object(shape=visual_object_info[0],
                                              **visual_object_info[1])
        return memory

    def add_rigid_general_object(self, name, shape, **object_params):
        """

        :param name:
        :param shape:
        :param object_params:
        :return:
        """
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.rigid_objects[name] = Cuboid(self.
                                              rigid_objects_client_instances,
                                              name,
                                              **object_params)
        elif shape == "static_cube":
            self.rigid_objects[name] = StaticCuboid(self.
                                                    rigid_objects_client_instances,
                                                    name,
                                                    **object_params)
        else:
            raise Exception("shape is not yet implemented")
        return

    def remove_general_object(self, name):
        """

        :param name:
        :return:
        """
        if name not in self.name_keys:
            raise Exception("name does not exists as key for scene objects")
        else:
            self.name_keys.remove(name)
        if name in self.rigid_objects.keys():
            self.rigid_objects[name].remove()
            del self.rigid_objects[name]
        elif name in self.visual_objects.keys():
            self.visual_objects[name].remove()
            del self.visual_objects[name]
        return

    def remove_everything(self):
        """

        :return:
        """
        current_objects = list(self.rigid_objects.keys()) + \
                           list(self.visual_objects.keys())
        for name in current_objects:
            self.remove_general_object(name)
        return

    def add_rigid_mesh_object(self, name, filename, **object_params):
        """

        :param name:
        :param filename:
        :param object_params:
        :return:
        """
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        self.rigid_objects[name] = MeshObject(self.rigid_objects_client_instances,
                                              name,
                                              filename, **object_params)
        return

    def add_silhoutte_general_object(self, name, shape, **object_params):
        """

        :param name:
        :param shape:
        :param object_params:
        :return:
        """
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.visual_objects[name] = SCuboid(
                self.visual_object_client_instances, name, **object_params)

            if self.observation_mode == "cameras":
                self.update_goal_image()
        elif shape == "sphere":
            self.visual_objects[name] = SSphere(
                self.visual_object_client_instances, name, **object_params)
            if self.observation_mode == "cameras":
                self.update_goal_image()
        else:
            raise Exception("shape is not implemented yet")
        return

    def add_silhoutte_mesh_object(self, name, filename, **object_params):
        """

        :param name:
        :param filename:
        :param object_params:
        :return:
        """
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        self.visual_objects[name] = SMeshObject(self.visual_object_client_instances,
                                                name,
                                                filename, **object_params)
        return

    def finalize_stage(self):
        """

        :return:
        """
        self.stage_observations = StageObservations(self.rigid_objects,
                                                    self.visual_objects,
                                                    self.observation_mode,
                                                    self.normalize_observations)

    def select_observations(self, observation_keys):
        """

        :param observation_keys:
        :return:
        """
        self.stage_observations.reset_observation_keys()
        for key in observation_keys:
            self.stage_observations.add_observation(key)

    def get_full_state(self, state_type='list'):
        """

        :param state_type:
        :return:
        """
        if state_type == 'list':
            stage_state = []
        elif state_type == 'dict':
            stage_state = dict()
        else:
            raise Exception("type is not supported")
        for name in self.name_keys:
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
            elif name in self.visual_objects:
                object = self.visual_objects[name]
            else:
                raise Exception("possible error here")
            if state_type == 'list':
                stage_state.extend(object.get_state(state_type='list'))
            elif state_type == 'dict':
                stage_state[name] = object.get_state(state_type='dict')
        return stage_state

    def set_full_state(self, new_state):
        """

        :param new_state:
        :return:
        """
        #TODO: under the assumption that the new state has the same number of objects
        start = 0
        for name in self.name_keys:
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
                end = start + object.get_state_size()
                object.set_full_state(new_state[start:end])
            elif name in self.visual_objects:
                object = self.visual_objects[name]
                end = start + object.get_state_size()
                object.set_full_state(new_state[start:end])
            start = end
        # self.latest_full_state = new_state
        if self.observation_mode == "cameras":
            self.update_goal_image()
        #self.pybullet_client.stepSimulation()
        return

    def set_objects_pose(self, names, positions, orientations):
        """

        :param names:
        :param positions:
        :param orientations:
        :return:
        """
        for i in range(len(names)):
            name = names[i]
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
                object.set_pose(positions[i], orientations[i])
            elif name in self.visual_objects:
                object = self.visual_objects[name]
                object.set_pose(positions[i], orientations[i])
            else:
                raise Exception("Object {} doesnt exist".format(name))
        # self.latest_full_state = self.get_full_state()
        if self.observation_mode == "cameras":
            self.update_goal_image()
        #self.pybullet_client.stepSimulation()
        return

    def get_current_observations(self, helper_keys):
        """

        :param helper_keys:
        :return:
        """
        return self.stage_observations.get_current_observations(helper_keys)

    def get_observation_spaces(self):
        """

        :return:
        """
        return self.stage_observations.get_observation_spaces()

    def random_position(self, height_limits=(0.05, 0.15),
                        angle_limits=(-2 * math.pi, 2 * math.pi),
                        radius_limits=(0.0, 0.15),
                        allowed_section=np.array([[-0.5, -0.5, 0],
                                                  [0.5, 0.5, 0.5]])):
        """

        :param height_limits:
        :param angle_limits:
        :param radius_limits:
        :param allowed_section:
        :return:
        """
        satisfying_constraints = False
        while not satisfying_constraints:
            angle = np.random.uniform(*angle_limits)
            # for uniform sampling with respect to the disc area use scaling
            radial_distance = np.sqrt(np.random.uniform(radius_limits[0]**2,
                                                        radius_limits[1]**2))

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

    def legacy_random_position(self, height_limits=(0.05, 0.15),
                                angle_limits=(-2 * math.pi, 2 * math.pi),
                                radius_limits=(0.0, 0.15)):
        """

        :param height_limits:
        :param angle_limits:
        :param radius_limits:
        :return:
        """
        angle = np.random.uniform(*angle_limits)
        radial_distance = np.random.uniform(*radius_limits)

        if isinstance(height_limits, (int, float)):
            height_z = height_limits
        else:
            height_z = np.random.uniform(*height_limits)

        object_position = [
            radial_distance * math.cos(angle),
            radial_distance * math.sin(angle),
            height_z,
        ]

        return object_position

    def clear(self):
        """

        :return:
        """
        # self.latest_full_state = None
        # self.latest_observations = None
        return

    def get_current_object_keys(self):
        """

        :return:
        """
        return list(self.rigid_objects.keys()) +  \
               list(self.visual_objects.keys())

    def object_intervention(self, key, interventions_dict):
        """

        :param key:
        :param interventions_dict:
        :return:
        """
        if key in self.rigid_objects:
            object = self.rigid_objects[key]
        elif key in self.visual_objects:
            object = self.visual_objects[key]
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))
        # save the old state of the object before intervention
        object.apply_interventions(interventions_dict)
        if self.observation_mode == "cameras":
            self.update_goal_image()
        #self.pybullet_client.stepSimulation()
        return

    def get_current_scm_values(self):
        """

        :return:
        """
        #TODO: not a complete list yet of what we want to expose
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id is not None
        else:
            client = self._pybullet_client_full_id
        variable_params = dict()
        variable_params["floor_color"] = \
            pybullet.getVisualShapeData(self.floor_id,
                                      physicsClientId=client)[0][7][:3]
        variable_params["stage_color"] = \
            pybullet.getVisualShapeData(self.stage_id,
                                      physicsClientId=client)[0][7][:3]
        variable_params["stage_friction"] = \
            pybullet.getDynamicsInfo(self.stage_id, -1,
                                     physicsClientId=client)[1]
        variable_params["floor_friction"] = \
            pybullet.getDynamicsInfo(self.floor_id, -1,
                                     physicsClientId=client)[1]
        variable_params["gravity"] = \
            self.current_gravity
        variable_params.update(self.get_full_state(state_type='dict'))
        return variable_params

    def apply_interventions(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        for intervention in interventions_dict:
            if isinstance(interventions_dict[intervention], dict):
                self.object_intervention(intervention,
                                             interventions_dict[intervention])
            elif intervention == "floor_color":
                for client in self.visual_object_client_instances:
                    pybullet.changeVisualShape(
                        self.floor_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1),
                        physicsClientId=client
                        )
                for client in self.rigid_objects_client_instances:
                    pybullet.changeVisualShape(
                        self.floor_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1),
                        physicsClientId=client)
            elif intervention == "stage_color":
                for client in self.visual_object_client_instances:
                    pybullet.changeVisualShape(
                        self.stage_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1),
                        physicsClientId=client)
                for client in self.rigid_objects_client_instances:
                    pybullet.changeVisualShape(
                        self.stage_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1),
                        physicsClientId=client)
            elif intervention == "stage_friction":
                for client in self.rigid_objects_client_instances:
                    pybullet.changeDynamics(
                        bodyUniqueId=self.stage_id,
                        linkIndex=-1,
                        lateralFriction=interventions_dict[intervention],
                        physicsClientId=client)
            elif intervention == "floor_friction":
                for client in self.rigid_objects_client_instances:
                    pybullet.changeDynamics(
                        bodyUniqueId=self.floor_id,
                        linkIndex=-1,
                        lateralFriction=interventions_dict[intervention],
                        physicsClientId=client)
            elif intervention == "gravity":
                for client in self.rigid_objects_client_instances:
                    pybullet.setGravity(interventions_dict[
                                        intervention][0],
                                      interventions_dict[
                                        intervention][1],
                                       interventions_dict[
                                        intervention][2],
                                        physicsClientId=client)
                self.current_gravity = interventions_dict[intervention]
            else:
                raise Exception("The intervention on stage "
                                "is not supported yet")
        if self.observation_mode == "cameras":
            self.update_goal_image()
        # self.latest_full_state = self.get_full_state()
        return

    def get_object_full_state(self, key):
        if key in self.rigid_objects:
            return self.rigid_objects[key].get_state('dict')
        elif key in self.visual_objects:
            return self.visual_objects[key].get_state('dict')
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))

    def get_object_state(self, key, state_variable):
        if key in self.rigid_objects:
            return np.array(self.rigid_objects[key].get_variable_state(
                state_variable))
        elif key in self.visual_objects:
            return np.array(self.visual_objects[key].get_variable_state(
                state_variable))
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))

    def get_object(self, key):
        if key in self.rigid_objects:
            return self.rigid_objects[key]
        elif key in self.visual_objects:
            return self.visual_objects[key]
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))

    def are_blocks_colliding(self, block1, block2):
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if (contact[1] == block1.block_id and contact[2] == block2.block_id) or \
                    (contact[2] == block1.block_id and contact[1] == block2.block_id):
                return True
        return False

    def check_stage_free_of_colliding_blocks(self):
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if contact[1] > 3 and contact[2] > 3:
                return False
        return True

    def is_colliding_with_stage(self, block1):
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if (contact[1] == block1.block_id and contact[2] == self.stage_id) or \
                    (contact[2] == block1.block_id and contact[1] == self.stage_id):
                return True
        return False

    def is_colliding_with_floor(self, block1):
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if (contact[1] == block1.block_id and contact[2] == self.floor_id) or \
                    (contact[2] == block1.block_id and contact[1] == self.floor_id):
                return True
        return False

    def get_normal_interaction_force_between_blocks(self, block1, block2):
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if (contact[1] == block1.block_id and contact[2] == block2.block_id) or \
                    (contact[2] == block1.block_id and contact[1] == block2.block_id):
                return contact[9]*np.array(contact[7])
        return None

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None):
        self.stage_observations.add_observation(observation_key, lower_bound,
                                                upper_bound)

    def normalize_observation_for_key(self, observation, key):
        return self.stage_observations.normalize_observation_for_key(observation,
                                                                     key)

    def denormalize_observation_for_key(self, observation, key):
        return self.stage_observations.denormalize_observation_for_key(observation,
                                                                       key)

    def get_current_goal_image(self):
        return self.goal_image

    def update_goal_image(self):
        if self.goal_image_pybullet_instance is not None:
            current_state = \
                self.goal_image_pybullet_instance.get_observation(0, update_images=True)
            self.goal_image = self.stage_observations.get_current_goal_image(
                current_state)
            return
        else:
            raise Exception("goal image is not enabled")

    def check_feasiblity_of_stage(self):
        """
        This function checks the feasibility of the current state of the stage
        (i.e checks if any of the bodies in the simulation are in a penetration
        mode)
        Parameters
        ---------

        Returns
        -------
            feasibility_flag: bool
                A boolean indicating whether the stage is in a collision state
                or not.
        """
        for contact in pybullet.getContactPoints(
                physicsClientId=self.rigid_objects_client_instances[0]):
            if contact[8] < -0.08:
                return False
        #check if all the visual objects are within the bb og the available arena
        for visual_object in self.visual_objects:
            if get_intersection(self.visual_objects[visual_object].
                                   get_bounding_box(),
                                   self._get_stage_bb())/\
                    self.visual_objects[visual_object].get_volume() < 0.95:
                return False
        return True

    def _get_stage_bb(self):
        return (tuple(self.floor_inner_bounding_box[0]),
                tuple(self.floor_inner_bounding_box[1]))
