from causal_world.evaluation.protocol import Protocol
import numpy as np


class ObjectColorsSpaceB(Protocol):

    def __init__(self):
        """

        """
        super().__init__('object_colors_space_B')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.get_intervention_space_b()
            color = None
            for rigid_object in self.env.get_task()._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'color' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    if color is None:
                        color = np.random.uniform(
                            intervention_space[rigid_object]['color'][0],
                            intervention_space[rigid_object]['color'][1])
                    intervention_dict[rigid_object]['color'] = color
            return intervention_dict
        else:
            return None
