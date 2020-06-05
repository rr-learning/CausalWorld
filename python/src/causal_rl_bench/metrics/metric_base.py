class BaseMetric(object):
    def __init__(self, name):
        """

        :param name:
        """
        self.name = name
        return

    def process_episode(self, episode_obj):
        """

        :param episode_obj:
        :return:
        """
        raise Exception("not implemendted yet")

    def get_metric_score(self):
        """

        :return:
        """
        raise Exception("not implemendted yet")
