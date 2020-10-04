class BaseMetric(object):

    def __init__(self, name):
        """
        The metric base to be used for any metric to calculate over the
        episodes evaluated.

        :param name: (str) metric name.
        """
        self.name = name
        return

    def process_episode(self, episode_obj):
        """
        Processes the episode to calculate the scores out of it.

        :param episode_obj: (causal_world.loggers.Episode) episode to process
                                                           and calculate its metric.
        :return:
        """
        raise Exception("not implemendted yet")

    def get_metric_score(self):
        """

        :return: (float) the metric score calculated so far.
        """
        raise Exception("not implemendted yet")

    def reset(self):
        """
        resets the metric calculation of episodes.

        :return:
        """
