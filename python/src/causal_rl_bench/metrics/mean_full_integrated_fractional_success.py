from causal_rl_bench.metrics.metric_base import BaseMetric


class MeanFullIntegratedFractionalSuccess(BaseMetric):

    def __init__(self):
        super(MeanFullIntegratedFractionalSuccess,
              self).__init__(name='mean_full_integrated_fractional_success')
        self.accumulated_success = 0
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
        self.accumulated_success += in_episode_accumulated_success / len(
            episode_obj.infos)

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
