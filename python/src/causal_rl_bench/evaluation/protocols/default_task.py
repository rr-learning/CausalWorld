from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class DefaultTask(Protocol):

    def __init__(self):
        super().__init__('default_task')

    def get_intervention(self, episode, timestep):
        return None
