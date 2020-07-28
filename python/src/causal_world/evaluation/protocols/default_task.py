from causal_world.evaluation.protocol import Protocol
import numpy as np


class DefaultTask(Protocol):

    def __init__(self):
        """

        """
        super().__init__('default_task')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:
        :return:
        """
        return None
