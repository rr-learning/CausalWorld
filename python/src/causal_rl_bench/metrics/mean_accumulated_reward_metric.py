from causal_rl_bench.metrics.metric_base import BaseMetric


class MeanAccumulatedRewardMetric(BaseMetric):
    def __init__(self):
        super(MeanAccumulatedRewardMetric, self).__init__(name='mean_accumulated_reward_rate')
        self.accumulated_reward = 0
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        """

        :param episode_obj:
        :return:
        """
        self.total_number_of_episodes += 1
        for rew in episode_obj.rewards:
            self.accumulated_reward += rew

    def get_metric_score(self):
        """

        :return:
        """
        return self.accumulated_reward / float(self.total_number_of_episodes)
