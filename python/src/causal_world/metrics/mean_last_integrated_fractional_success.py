from causal_world.metrics.metric_base import BaseMetric
import numpy as np


class MeanLastIntegratedFractionalSuccess(BaseMetric):

    def __init__(self):
        '''
        This metric reports the mean of the fractional success accumulated over the last 20 timesteps
        '''
        super(MeanLastIntegratedFractionalSuccess,
              self).__init__(name='last_integrated_fractional_success')
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
        for index in range(20):
            in_episode_accumulated_success += episode_obj.infos[-index][
                'fractional_success']
        self.per_episode_scores.append(in_episode_accumulated_success / 20)

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
