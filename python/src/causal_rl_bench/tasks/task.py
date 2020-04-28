class Task(object):
    def __init__(self):
        pass

    def get_scene_objects(self):
        raise NotImplementedError()

    def get_counterfactual_variant(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError

    def get_structured_observation_space(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError()

    def reset_task(self):
        raise NotImplementedError()
