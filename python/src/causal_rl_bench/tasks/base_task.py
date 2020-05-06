class BaseTask(object):
    def __init__(self):
        pass

    def init_task(self, robot, stage):
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

    def is_done(self):
        raise NotImplementedError()

    def do_random_intervention(self):
        raise NotImplementedError()

    def do_intervention(self, **kwargs):
        raise NotImplementedError()

