from causal_world.metrics.metric_base import BaseMetric


class MeanAccumulatedRewardMetric(BaseMetric):

    def __init__(self):
        """
        The MeanAccumulatedRewardMetric to be used to calculate the mean
        accumlated reward over all episodes processed.
        """
        super(MeanAccumulatedRewardMetric,
              self).__init__(name='mean_accumulated_reward_rate')
        self.accumulated_reward = 0
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
        for rew in episode_obj.rewards:
            self.accumulated_reward += rew

    def get_metric_score(self):
        """

        :return: (float) the metric score calculated so far.
        """
        return self.accumulated_reward / float(self.total_number_of_episodes)

    def reset(self):
        """
        resets the metric calculation of episodes.

        :return:
        """
        self.accumulated_reward = 0
        self.total_number_of_episodes = 0
