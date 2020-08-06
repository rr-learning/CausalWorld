from causal_world.evaluation.protocol import Protocol
import numpy as np


class ObjectMassesSpaceB(Protocol):

    def __init__(self):
        """

        """
        super().__init__('object_masses_space_B')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.get_intervention_space_b()
            mass = None
            for rigid_object in self.env.get_task()._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'mass' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    if mass is None:
                        mass = np.random.uniform(
                            intervention_space[rigid_object]['mass'][0],
                            intervention_space[rigid_object]['mass'][1])
                    intervention_dict[rigid_object]['mass'] = mass
            return intervention_dict
        else:
            return None
