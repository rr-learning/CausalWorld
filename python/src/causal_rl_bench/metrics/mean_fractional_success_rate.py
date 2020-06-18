from causal_rl_bench.metrics.metric_base import BaseMetric


class MeanFractionalSuccessRate(BaseMetric):
    def __init__(self):
        super(MeanFractionalSuccessRate, self).__init__(name='mean_fractional_success_rate')
        self.accumulated_success = 0
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        """

        :param episode_obj:
        :return:
        """
        self.total_number_of_episodes += 1
        self.accumulated_success += episode_obj.infos[-1]['fractional_reward']

    def get_metric_score(self):
        """

        :return:
        """
        return self.accumulated_success / float(self.total_number_of_episodes)

    def reset(self):
        """
        :return:
        """
        self.accumulated_success = 0
        self.total_number_of_episodes = 0
