from causal_rl_bench.loggers.data_recorder import DataRecorder


class BaseMetric(object):
    def __init__(self, name):
        self.name = name
        return

    def process_episode(self, episode_obj):
        raise Exception("not implemendted yet")

    def get_metric_score(self):
        raise Exception("not implemendted yet")
