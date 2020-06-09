from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class DefaultTask(Protocol):
    def __init__(self):
        self.name = 'default_task'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, env, episode, timestep):
        return None
