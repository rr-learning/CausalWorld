from causal_rl_bench.envs.scene.observations import StageObservations
from causal_rl_bench.envs.scene.objects import Cuboid
from causal_rl_bench.envs.scene.silhouette import SCuboid
import math
import numpy as np


class Stage(object):
    def __init__(self, pybullet_client, observation_mode,
                 normalize_observations=True,
                 goal_image_pybullet_instance=None):
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.observation_mode = observation_mode
        self.pybullet_client = pybullet_client
        self.normalize_observations = normalize_observations
        self.stage_observations = None
        self.latest_full_state = None
        self.latest_observations = None
        self.goal_image_pybullet_instance = goal_image_pybullet_instance
        if self.goal_image_pybullet_instance is not None:
            self.goal_image_visual_objects = dict()
            self.goal_image_pybullet_client = \
                self.goal_image_pybullet_instance._p
            self.goal_image = None
        self.name_keys = []
        return

    def add_rigid_general_object(self, name, shape, **object_params):
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.rigid_objects[name] = Cuboid(self.pybullet_client, name,
                                              **object_params)
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

    def get_full_state(self):
        stage_state = []
        for name in self.name_keys:
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
            elif name in self.visual_objects:
                object = self.visual_objects[name]
            stage_state.extend(object.get_state(state_type='list'))
        self.latest_full_state = stage_state
        return self.latest_full_state

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
        return

    def get_current_observations(self, helper_keys):
        self.latest_observations = \
            self.stage_observations.get_current_observations(helper_keys)
        return self.latest_observations

    def get_observation_spaces(self):
        return self.stage_observations.get_observation_spaces()

    def random_position(self, height_limits=(0.05, 0.15),
                        angle_limits=(-2 * math.pi, 2 * math.pi),
                        radius_limits=(0.0, 0.15)):

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

        return object_position

    def clear(self):
        self.latest_full_state = None
        self.latest_observations = None

    def get_current_object_keys(self):
        return list(self.rigid_objects.keys()) +  \
               list(self.visual_objects.keys())

    def object_intervention(self, key, interventions_dict):
        if key in self.rigid_objects:
            object = self.rigid_objects[key]
            object.set_state(interventions_dict)
        elif key in self.visual_objects:
            object = self.visual_objects[key]
            object.set_state(interventions_dict)
            if self.goal_image_pybullet_instance is not None:
                goal_image_object = self.goal_image_visual_objects[key]
                goal_image_object.set_state(interventions_dict)
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))
        if self.goal_image_pybullet_instance is not None:
            self.update_goal_image()
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
            return np.array(self.rigid_objects[key].get_variable_state(state_variable))
        elif key in self.visual_objects:
            return np.array(self.visual_objects[key].get_variable_state(state_variable))
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))

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

