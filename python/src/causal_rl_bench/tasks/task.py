from counterfactual.python.src.causal_rl_bench.envs.scene.objects import SceneObjects


class Task(object):
    def __init__(self):
        self.scene_objects = SceneObjects()

    def get_scene_objects(self):
        return self.scene_objects

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
