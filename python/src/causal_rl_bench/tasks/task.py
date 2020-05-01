class Task(object):
    def __init__(self):
        pass

    def init_task(self, robot, stage):
        raise NotImplementedError()

    def reset_to_counterfactual_variant(self, **kwargs):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError()

    def reset_task(self):
        raise NotImplementedError()

    def filter_observations(self, robot_observations_dict,
                            stage_observations_dict):
        raise NotImplementedError()

    def reset_scene_objects(self):
        raise NotImplementedError()

    def get_task_params(self):
        raise NotImplementedError()
