from causal_rl_bench.envs.scene.observations import StageObservations


class Stage(object):
    def __init__(self, observations_mode):
        self.rigid_objects = []
        self.visual_objects = []
        self.stage_observations = StageObservations(observation_mode)
        pass

    def add_rigid_general_object(self, key, shape, position, orientation, size,
                                 fixed=False):
        raise Exception(" Not implemented")

    def add_rigid_mesh_object(self, key, file, position, orientation,
                              fixed=False):
        raise Exception(" Not implemented")

    def add_silhoutte_general_object(self, key, shape, position, orientation, size):
        raise Exception(" Not implemented")

    def add_silhoutte_mesh_object(self, key, file, position, orientation, size):
        raise Exception(" Not implemented")

    def get_full_state(self):
        raise Exception(" Not implemented")

    def set_full_state(self):
        raise Exception(" Not implemented")