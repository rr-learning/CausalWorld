from causal_world.evaluation.protocol import Protocol
import numpy as np


class ObjectSizesOOD(Protocol):

    def __init__(self):
        """

        """
        super().__init__('object_sizes_ood')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task._testing_intervention_spaces
            for rigid_object in self.env._task._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'size' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['size'] = \
                        np.random.uniform(intervention_space[rigid_object]['size'][0],
                                          intervention_space[rigid_object]['size'][1])
            return intervention_dict
        else:
            return None