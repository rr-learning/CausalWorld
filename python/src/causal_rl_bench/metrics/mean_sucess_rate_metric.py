from causal_rl_bench.metrics.metric_base import BaseMetric


class MeanSuccessRateMetric(BaseMetric):
    def __init__(self):
        super(MeanSuccessRateMetric, self).__init__(name='mean_success_rate')
        self.success_times = 0
        self.total_number_of_episodes = 0
        return

    def process_episode(self, episode_obj):
        self.total_number_of_episodes += 1
        for done_signal in episode_obj.dones:
            if done_signal:
                self.success_times += 1
                return

    def get_metric_score(self):
        return self.success_times / float(self.total_number_of_episodes)
