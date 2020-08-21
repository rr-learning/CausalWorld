from causal_world.evaluation.protocol import Protocol
import numpy as np


class Protocol0(Protocol):

    def __init__(self):
        """
        DefaultTask
        """
        super().__init__('P0')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:
        :return:
        """
        return None
