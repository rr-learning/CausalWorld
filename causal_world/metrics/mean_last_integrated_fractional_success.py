from causal_world.metrics.metric_base import BaseMetric
import numpy as np


class MeanLastIntegratedFractionalSuccess(BaseMetric):

    def __init__(self):
        """
        The MeanLastIntegratedFractionalSuccess to be used to calculate the mean
        over last 20 fractional successes over all episodes processed.
        """
        super(MeanLastIntegratedFractionalSuccess,
              self).__init__(name='last_integrated_fractional_success')
        self.per_episode_scores = []
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        """

        Processes the episode to calculate the scores out of it.

        :param episode_obj: (causal_world.loggers.Episode) episode to process
                                                           and calculate its metric.
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

        :return: (tuple) the mean of the metric score,
                         the std of the metric score.
        """
        return (np.mean(self.per_episode_scores), np.std(self.per_episode_scores))

    def reset(self):
        """
        resets the metric calculation of episodes.

        :return:
        """
        self.per_episode_scores = []
        self.total_number_of_episodes = 0
