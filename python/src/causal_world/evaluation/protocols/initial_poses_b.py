from causal_world.evaluation.protocol import Protocol
import numpy as np


class InitialPosesSpaceB(Protocol):

    def __init__(self):
        """

        """
        super().__init__('initial_poses_space_B')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.get_intervention_space_b()
            for rigid_object in self.env.get_task()._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'cartesian_position' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['cartesian_position'] = \
                        np.random.uniform(intervention_space[rigid_object]['cartesian_position'][0],
                                          intervention_space[rigid_object]['cartesian_position'][1])
            return intervention_dict
        else:
            return None
