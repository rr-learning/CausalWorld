from causal_world.evaluation.protocols.protocol import ProtocolBase
import numpy as np


class FullyRandomProtocol(ProtocolBase):

    def __init__(self, name, variable_space='a_b'):
        """

        """
        super().__init__(name)
        self._variable_space = variable_space

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            if self._variable_space == 'a_b':
                intervention_space = self.env.get_intervention_space_a_b()
            elif self._variable_space == 'a':
                intervention_space = self.env.get_intervention_space_a()
            elif self._variable_space == 'b':
                intervention_space = self.env.get_intervention_space_b()
            interventions_dict = dict()
            intervene_on_size = np.random.choice([0, 1], p=[0.5, 0.5])
            intervene_on_joint_positions = np.random.choice([0, 1],
                                                            p=[0.9, 0.1])
            for variable in intervention_space:
                if isinstance(intervention_space[variable], dict):
                    interventions_dict[variable] = dict()
                    for subvariable_name in intervention_space[
                        variable]:
                        if subvariable_name == 'cylindrical_position' and \
                                intervene_on_size:
                            continue
                        if subvariable_name == 'size' and not intervene_on_size:
                            continue
                        interventions_dict[variable][subvariable_name] = \
                            np.random.uniform(
                                intervention_space
                                [variable][subvariable_name][0],
                                intervention_space
                                [variable][subvariable_name][1])
                else:
                    if not intervene_on_joint_positions and variable == 'joint_positions':
                        continue
                    interventions_dict[variable] = np.random.uniform(
                        intervention_space[variable][0],
                        intervention_space[variable][1])
            return interventions_dict
        else:
            return None

    def _init_protocol_helper(self):
        if self._variable_space == 'a_b':
            self.env.set_intervention_space(variables_space='space_a_b')
        elif self._variable_space == 'a':
            self.env.set_intervention_space(variables_space='space_a')
        elif self._variable_space == 'b':
            self.env.set_intervention_space(variables_space='space_b')
        return
