from causal_world.metrics.metric_base import BaseMetric
import numpy as np


class MeanFullIntegratedFractionalSuccess(BaseMetric):

    def __init__(self):
        super(MeanFullIntegratedFractionalSuccess,
              self).__init__(name='full_integrated_fractional_success')
        self.per_episode_scores = []
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        """

        :param episode_obj:
        :return:
        """
        self.total_number_of_episodes += 1
        in_episode_accumulated_success = 0.0
        for info in episode_obj.infos:
            in_episode_accumulated_success += info['fractional_success']
        self.per_episode_scores.append(in_episode_accumulated_success / len(
            episode_obj.infos))

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
