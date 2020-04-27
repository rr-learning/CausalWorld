class StageObservations(object):
    def __init__(self, observation_mode, normalize_observations=True):
        pass

    def set_observation_space(self, key):
        raise Exception(" Not implemented")

    def get_observation_space(self, key):
        raise Exception(" Not implemented")

    def get_observation_spaces(self):
        raise Exception(" Not implemented")

    def is_normalized(self, key):
        raise Exception(" Not implemented")

    def normalize_observation(self, key):
        raise Exception(" Not implemented")

    def denormalize_observation(self, key):
        raise Exception(" Not implemented")

    def satisfy_constraints(self):
        raise Exception(" Not implemented")

    def clip_observation(self, key):
        raise Exception(" Not implemented")



