from causal_world.evaluation.protocol import Protocol
import numpy as np


class Protocol9(Protocol):

    def __init__(self):
        """
        GoalPosesInitialPosesMassesFloorFrictionSpaceB
        """
        super().__init__('P9')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """

        if timestep == 0:
            intervention_dict = self.env.sample_new_goal(level=None)
            intervention_space = self.env.get_intervention_space_b()
            mass = None
            for rigid_object in self.env.get_task()._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        rigid_object != 'obstacle' and \
                        'cylindrical_position' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['cylindrical_position'] = \
                        np.random.uniform(intervention_space[rigid_object]['cylindrical_position'][0],
                                          intervention_space[rigid_object]['cylindrical_position'][1])
                    height = self.env.get_task()._stage._rigid_objects[rigid_object].get_variable_state('size')[2]
                    intervention_dict[rigid_object]['cylindrical_position'][2] = height/2
                    if mass is None:
                        mass = np.random.uniform(
                            intervention_space[rigid_object]['mass'][0],
                            intervention_space[rigid_object]['mass'][1])
                    intervention_dict[rigid_object]['mass'] = mass
                    floor_friction = np.random.uniform(
                        intervention_space['floor_friction'][0],
                        intervention_space['floor_friction'][1])
                    intervention_dict['floor_friction'] = floor_friction
            return intervention_dict
        else:
            return None

    def _init_protocol_helper(self):
        self.env.set_intervention_space(variables_space='space_b')
        return
