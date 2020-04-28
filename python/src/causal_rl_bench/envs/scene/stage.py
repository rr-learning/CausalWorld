from causal_rl_bench.envs.scene.observations import StageObservations
from causal_rl_bench.envs.scene.objects import Cuboid
from causal_rl_bench.envs.scene.silhouette import SCuboid


class Stage(object):
    def __init__(self, observation_mode,
                 normalize_observations=True):
        self.rigid_objects = dict()
        self.visual_objects = dict()
        self.observation_mode = observation_mode
        self.normalize_observations = normalize_observations
        self.stage_observations = None
        self.latest_full_state = None
        return

    def add_rigid_general_object(self, name, shape, object_params):
        if shape == "cube":
            self.rigid_objects[name] = Cuboid(name, **object_params)
        return

    def add_rigid_mesh_object(self, name, file, object_params):
        raise Exception(" Not implemented")

    def add_silhoutte_general_object(self, name, shape, object_params):
        if shape == "cube":
            self.visual_objects[name] = SCuboid(name, **object_params)
        return

    def add_silhoutte_mesh_object(self, name, file, object_params):
        raise Exception(" Not implemented")

    def finalize_stage(self):
        self.stage_observations = StageObservations(self.observation_mode,
                                                    self.normalize_observations,
                                                    self.rigid_objects.values(),
                                                    self.visual_objects.values())

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
