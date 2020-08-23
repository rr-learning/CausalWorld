from causal_world.evaluation.protocol import Protocol
import numpy as np


class Protocol4(Protocol):

    def __init__(self):
        """
        InitialPosesSpaceA
        """
        super().__init__('P4')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.get_intervention_space_a()
            for rigid_object in self.env.get_task()._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        rigid_object != 'obstacle' and \
                        'cylindrical_position' in intervention_space[rigid_object]:
                    if self.env.get_task()._stage._rigid_objects[rigid_object].is_not_fixed():
                        intervention_dict[rigid_object] = dict()
                        intervention_dict[rigid_object]['cylindrical_position'] = \
                            np.random.uniform(intervention_space[rigid_object]['cylindrical_position'][0],
                                              intervention_space[rigid_object]['cylindrical_position'][1])
                        height = self.env.get_task()._stage._rigid_objects[rigid_object].get_variable_state('size')[2]
                        intervention_dict[rigid_object]['cylindrical_position'][2] = height/2
            return intervention_dict
        else:
            return None
