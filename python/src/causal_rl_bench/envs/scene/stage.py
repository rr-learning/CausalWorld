"""
causal_rl_bench/envs/scene/stage.py
===================================
"""
from causal_rl_bench.envs.scene.observations import StageObservations
from causal_rl_bench.envs.scene.objects import Cuboid, StaticCuboid
from causal_rl_bench.envs.scene.silhouette import SCuboid, SSphere
import math
import numpy as np


class Stage(object):
    """This class holds everything in the arena including the tools used
    for the goal specified, setting object, creating of objects, removing of
    objects, checking for collisions, all of this is done here"""
    def __init__(self, pybullet_client, observation_mode,
                 normalize_observations=True,
                 goal_image_pybullet_instance=None):
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.observation_mode = observation_mode
        self.pybullet_client = pybullet_client
        #TODO: move the ids from here
        self.floor_id = 2
        self.stage_id = 3
        self.normalize_observations = normalize_observations
        self.stage_observations = None
        self.latest_full_state = None
        self.latest_observations = None
        self.goal_image_pybullet_instance = goal_image_pybullet_instance
        self.floor_height = 0.01
        self.floor_inner_bounding_box = np.array(
            [[-0.15, -0.15, self.floor_height], [0.15, 0.15, 0.2]])
        if self.goal_image_pybullet_instance is not None:
            self.goal_image_visual_objects = dict()
            self.goal_image_pybullet_client = \
                self.goal_image_pybullet_instance._p
            self.goal_image = None
        self.name_keys = []
        # self.arena_interventions({'floor_color': np.array([1, 0.5, 1])})
        # self.arena_interventions({'stage_color': np.array([0.3, 0.5, 1])})
        # self.arena_interventions({'stage_friction': 0.8})
        # print(self.get_current_variables_values())
        return

    def add_rigid_general_object(self, name, shape, **object_params):
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.rigid_objects[name] = Cuboid(self.pybullet_client, name,
                                              **object_params)
        elif shape == "static_cube":
            self.rigid_objects[name] = StaticCuboid(self.pybullet_client, name,
                                                    **object_params)
        else:
            raise Exception("shape is not yet implemented")
        return

    def remove_general_object(self, name):
        if name not in self.name_keys:
            raise Exception("name does not exists as key for scene objects")
        else:
            self.name_keys.remove(name)
        if name in self.rigid_objects.keys():
            block_id = self.rigid_objects[name].block_id
            del self.rigid_objects[name]
            self.pybullet_client.removeBody(block_id)
        elif name in self.visual_objects.keys():
            block_id = self.visual_objects[name].block_id
            del self.visual_objects[name]
            self.pybullet_client.removeBody(block_id)
        return

    def remove_everything(self):
        current_objects = list(self.rigid_objects.keys()) + list(self.visual_objects.keys())
        for name in current_objects:
            self.remove_general_object(name)
        return

    def add_rigid_mesh_object(self, name, file, **object_params):
        raise Exception(" Not implemented")

    def add_silhoutte_general_object(self, name, shape, **object_params):
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.visual_objects[name] = SCuboid(self.pybullet_client, name,
                                                **object_params)
            if self.goal_image_pybullet_instance is not None:
                self.goal_image_visual_objects[name] = SCuboid(
                    self.goal_image_pybullet_client, name, **object_params)
        elif shape == "sphere":
            self.visual_objects[name] = SSphere(self.pybullet_client, name,
                                                **object_params)
            if self.goal_image_pybullet_instance is not None:
                self.goal_image_visual_objects[name] = SSphere(
                    self.goal_image_pybullet_client, name, **object_params)
        else:
            raise Exception("shape is not implemented yet")
        return

    def add_silhoutte_mesh_object(self, name, file, **object_params):
        raise Exception(" Not implemented")

    def finalize_stage(self):
        self.stage_observations = StageObservations(self.rigid_objects.values(),
                                                    self.visual_objects.values(),
                                                    self.observation_mode,
                                                    self.normalize_observations)

    def select_observations(self, observation_keys):
        self.stage_observations.reset_observation_keys()
        for key in observation_keys:
            self.stage_observations.add_observation(key)

    def get_full_state(self, state_type='list'):
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
                if self.goal_image_pybullet_instance is not None:
                    goal_image_object = self.goal_image_visual_objects[name]
                    goal_image_object.set_full_state(new_state[start:end])
            start = end
        self.latest_full_state = new_state
        if self.goal_image_pybullet_instance is not None:
            self.update_goal_image()
        self.pybullet_client.stepSimulation()
        return

    def set_objects_pose(self, names, positions, orientations):
        for i in range(len(names)):
            name = names[i]
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
                object.set_pose(positions[i], orientations[i])
            elif name in self.visual_objects:
                object = self.visual_objects[name]
                object.set_pose(positions[i], orientations[i])
                if self.goal_image_pybullet_instance is not None:
                    goal_image_object = self.goal_image_visual_objects[name]
                    goal_image_object.set_pose(positions[i], orientations[i])
            else:
                raise Exception("Object {} doesnt exist".format(name))
        self.latest_full_state = self.get_full_state()
        if self.goal_image_pybullet_instance is not None:
            self.update_goal_image()
        self.pybullet_client.stepSimulation()
        return

    def get_current_observations(self, helper_keys):
        self.latest_observations = \
            self.stage_observations.get_current_observations(helper_keys)
        return self.latest_observations

    def get_observation_spaces(self):
        return self.stage_observations.get_observation_spaces()

    def random_position(self, height_limits=(0.05, 0.15),
                        angle_limits=(-2 * math.pi, 2 * math.pi),
                        radius_limits=(0.0, 0.15),
                        allowed_section=np.array([[-0.5, -0.5, 0], [0.5, 0.5, 0.5]])):

        satisfying_constraints = False
        while not satisfying_constraints:
            angle = np.random.uniform(*angle_limits)
            # for uniform sampling with respect to the disc area use scaling
            radial_distance = np.sqrt(np.random.uniform(radius_limits[0]**2, radius_limits[1]**2))

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
        self.latest_full_state = None
        self.latest_observations = None

    def get_current_object_keys(self):
        return list(self.rigid_objects.keys()) +  \
               list(self.visual_objects.keys())

    def object_intervention(self, key, interventions_dict):
        success_intervention = True
        if key in self.rigid_objects:
            object = self.rigid_objects[key]
        elif key in self.visual_objects:
            object = self.visual_objects[key]
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))
        # save the old state of the object before intervention
        old_state = object.get_state(state_type='list')
        object.set_state(interventions_dict)
        if not self.check_feasiblity_of_stage():
            object.set_full_state(old_state)
            success_intervention = False
        if key in self.visual_objects and \
                self.goal_image_pybullet_instance is not None:
            #TODO: under the impression that an intervention of visuals are always
            #feasible
            goal_image_object = self.goal_image_visual_objects[key]
            goal_image_object.set_state(interventions_dict)
            self.update_goal_image()
        self.pybullet_client.stepSimulation()
        return success_intervention

    def get_current_variables_values(self):
        #TODO: not a complete list yet of what we want to expose
        variable_params = dict()
        variable_params["floor_color"] = \
            self.pybullet_client.getVisualShapeData(self.floor_id)[0][7][:3]
        variable_params["stage_color"] = \
            self.pybullet_client.getVisualShapeData(self.stage_id)[0][7][:3]
        variable_params["stage_friction"] = \
            self.pybullet_client.getDynamicsInfo(self.stage_id, -1)[1]
        variable_params["floor_friction"] = \
            self.pybullet_client.getDynamicsInfo(self.floor_id, -1)[1]
        variable_params.update(self.get_full_state(state_type='dict'))
        return variable_params

    def apply_interventions(self, interventions_dict):
        success_intervention = True
        for intervention in interventions_dict:
            if isinstance(interventions_dict[intervention], dict):
                success_intervention = \
                    self.object_intervention(intervention,
                                             interventions_dict[intervention])
            elif intervention == "floor_color":
                self.pybullet_client.changeVisualShape(
                    self.floor_id, -1, rgbaColor=np.append(
                        interventions_dict[intervention], 1))
                if self.goal_image_pybullet_instance is not None:
                    self.goal_image_pybullet_instance.changeVisualShape(
                        self.floor_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1))
            elif intervention == "stage_color":
                self.pybullet_client.changeVisualShape(
                    self.stage_id, -1, rgbaColor=np.append(
                        interventions_dict[intervention], 1))
                if self.goal_image_pybullet_instance is not None:
                    self.goal_image_pybullet_instance.changeVisualShape(
                        self.stage_id, -1, rgbaColor=np.append(
                            interventions_dict[intervention], 1))
            elif intervention == "stage_friction":
                self.pybullet_client.changeDynamics(
                    bodyUniqueId=self.stage_id,
                    linkIndex=-1,
                    lateralFriction=interventions_dict[intervention],
                )
            elif intervention == "floor_friction":
                self.pybullet_client.changeDynamics(
                    bodyUniqueId=self.floor_id,
                    linkIndex=-1,
                    lateralFriction=interventions_dict[intervention],
                )
            else:
                raise Exception("The intervention on stage "
                                "is not supported yet")
        return success_intervention

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
            return np.array(self.rigid_objects[key].get_variable_state(state_variable))
        elif key in self.visual_objects:
            return np.array(self.visual_objects[key].get_variable_state(state_variable))
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
        for contact in self.pybullet_client.getContactPoints():
            if (contact[1] == block1.block_id and contact[2] == block2.block_id) or \
                    (contact[2] == block1.block_id and contact[1] == block2.block_id):
                return True
        return False

    def check_stage_free_of_colliding_blocks(self):
        for contact in self.pybullet_client.getContactPoints():
            if contact[1] > 3 and contact[2] > 3:
                return False
        return True

    def is_colliding_with_stage(self, block1):
        for contact in self.pybullet_client.getContactPoints():
            if (contact[1] == block1.block_id and contact[2] == self.stage_id) or \
                    (contact[2] == block1.block_id and contact[1] == self.stage_id):
                return True
        return False

    def is_colliding_with_floor(self, block1):
        for contact in self.pybullet_client.getContactPoints():
            if (contact[1] == block1.block_id and contact[2] == self.floor_id) or \
                    (contact[2] == block1.block_id and contact[1] == self.floor_id):
                return True
        return False

    def get_normal_interaction_force_between_blocks(self, block1, block2):
        for contact in self.pybullet_client.getContactPoints():
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
        for contact in self.pybullet_client.getContactPoints():
            if contact[8] < -0.005:
                return False
        return True

