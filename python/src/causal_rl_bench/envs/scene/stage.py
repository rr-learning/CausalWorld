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
        self.latest_full_state = \
            self.stage_observations.get_current_observations()
        return self.latest_full_state

    def set_states(self, names, positions, orientations):
        for i in range(len(names)):
            name = names[i]
            if name in self.rigid_objects.keys():
                self.rigid_objects[name].set_state(positions[i],
                                                   orientations[i])
            if name in self.visual_objects.keys():
                self.visual_objects[name].set_state(positions[i],
                                                    orientations[i])
        return

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

    def clear_stage(self):
        # TODO: Need to remove rigid and visual objects here
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.stage_observations = None
        self.latest_full_state = None
        self.name_keys = []


    def clear(self):
        self.latest_full_state = None

    def select_observations(self, observation_keys):
        current_observations_keys = self.stage_observations.observations_keys
        for key in current_observations_keys:
            if key not in observation_keys:
                self.stage_observations.remove_observations([key])
