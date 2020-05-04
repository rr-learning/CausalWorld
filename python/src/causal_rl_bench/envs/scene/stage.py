from causal_rl_bench.envs.scene.observations import StageObservations
from causal_rl_bench.envs.scene.objects import Cuboid
from causal_rl_bench.envs.scene.silhouette import SCuboid
import math
import numpy as np


class Stage(object):
    def __init__(self, observation_mode,
                 normalize_observations=True):
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.observation_mode = observation_mode
        self.normalize_observations = normalize_observations
        self.stage_observations = None
        self.latest_full_state = None
        self.latest_observations = None
        self.name_keys = []
        return

    def add_rigid_general_object(self, name, shape, **object_params):
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.rigid_objects[name] = Cuboid(name, **object_params)
        return

    def add_rigid_mesh_object(self, name, file, **object_params):
        raise Exception(" Not implemented")

    def add_silhoutte_general_object(self, name, shape, **object_params):
        if name in self.name_keys:
            raise Exception("name already exists as key for scene objects")
        else:
            self.name_keys.append(name)
        if shape == "cube":
            self.visual_objects[name] = SCuboid(name, **object_params)
        return

    def add_silhoutte_mesh_object(self, name, file, **object_params):
        raise Exception(" Not implemented")

    def finalize_stage(self):
        self.stage_observations = StageObservations(self.rigid_objects.values(),
                                                    self.visual_objects.values(),
                                                    self.observation_mode,
                                                    self.normalize_observations)

    def get_full_state(self):
        stage_state = []
        for name in self.name_keys:
            if name in self.rigid_objects:
                object =  self.rigid_objects[name]
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
            elif name in self.visual_objects:
                object = self.visual_objects[name]
            end = start + object.get_state_size()
            object.set_full_state(new_state[start:end])
            start = end
        self.latest_full_state = new_state
        self.latest_observations = self.stage_observations.\
            get_current_observations()
        return self.latest_full_state

    def set_positions(self, object_names, positions_list, orientations_list):
        for i in range(len(object_names)):
            name = object_names[i]
            if name in self.rigid_objects:
                object = self.rigid_objects[name]
            elif name in self.visual_objects:
                object = self.visual_objects[name]
            else:
                raise Exception("Object {} doesnt exist".format(name))
            object.set_position(positions_list[i], orientations_list[i])
        self.latest_full_state = self.get_full_state()
        return

    def get_current_observations(self):
        self.latest_observations = \
            self.stage_observations.get_current_observations()
        return self.latest_observations

    def get_observation_spaces(self):
        return self.stage_observations.get_observation_spaces()

    def random_position(self, height_limits=(0.05, 0.15),
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

    def select_observations(self, observation_keys):
        current_observations_keys = self.stage_observations.observations_keys
        for key in current_observations_keys:
            if key not in observation_keys:
                self.stage_observations.remove_observations([key])

    def get_current_object_keys(self):
        return list(self.rigid_objects.keys()) +  \
               list(self.visual_objects.keys())

    def object_intervention(self, key, interventions_dict):
        if key in self.rigid_objects:
            object = self.rigid_objects[key]
        elif key in self.visual_objects:
            object = self.visual_objects[key]
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))
        object.set_state(interventions_dict)
        return

    def get_object_state(self, key):
        if key in self.rigid_objects:
            return self.rigid_objects[key].get_state('dict')
        elif key in self.visual_objects:
            return self.visual_objects[key].get_state('dict')
        else:
            raise Exception("The key {} passed doesn't exist in the stage yet"
                            .format(key))
