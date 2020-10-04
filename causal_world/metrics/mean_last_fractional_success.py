from causal_world.metrics.metric_base import BaseMetric
import numpy as np


class MeanLastFractionalSuccess(BaseMetric):

    def __init__(self):
        """
        The MeanLastFractionalSuccess to be used to calculate the mean
        last fractional success over all episodes processed.
        """
        super(MeanLastFractionalSuccess,
              self).__init__(name='last_fractional_success')
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
        self.per_episode_scores.append(episode_obj.infos[-1]['fractional_success'])

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
