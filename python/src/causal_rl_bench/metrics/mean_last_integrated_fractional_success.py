from causal_rl_bench.metrics.metric_base import BaseMetric


class MeanLastIntegratedFractionalSuccess(BaseMetric):
    def __init__(self):
        '''
        This metric reports the mean of the fractional success accumulated over the last 20 timesteps
        '''
        super(MeanLastIntegratedFractionalSuccess, self).__init__(name='mean_last_integrated_fractional_success')
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
        for index in range(20):
            in_episode_accumulated_success += episode_obj.infos[-index]['fractional_reward']
        self.accumulated_success += in_episode_accumulated_success / 20

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
