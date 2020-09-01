from causal_world.metrics.metric_base import BaseMetric
import numpy as np


class MeanLastFractionalSuccess(BaseMetric):

    def __init__(self):
        super(MeanLastFractionalSuccess,
              self).__init__(name='last_fractional_success')
        self.per_episode_scores = []
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        """

        :param episode_obj:
        :return:
        """
        self.total_number_of_episodes += 1
        self.per_episode_scores.append(episode_obj.infos[-1]['fractional_success'])

    def get_metric_score(self):
        """

        :return:
        """
        return (np.mean(self.per_episode_scores), np.std(self.per_episode_scores))

    def reset(self):
        """
        :return:
        """
        self.per_episode_scores = []
        self.total_number_of_episodes = 0
