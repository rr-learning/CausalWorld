from causal_world.evaluation.protocols.protocol import ProtocolBase
import numpy as np
import re


class ProtocolGenerator(ProtocolBase):

    def __init__(self, name, first_level_regex,
                 second_level_regex,
                 variable_space='a_b'):
        """

        """
        super().__init__(name)
        self._first_level_regex = first_level_regex
        self._second_level_regex = second_level_regex
        self._variable_space = variable_space

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            intervention_dict = dict()
            if self._variable_space == 'a_b':
                intervention_space = self.env.get_intervention_space_a_b()
            elif self._variable_space == 'a':
                intervention_space = self.env.get_intervention_space_a()
            elif self._variable_space == 'b':
                intervention_space = self.env.get_intervention_space_b()
            for variable in intervention_space:
                if re.fullmatch(self._first_level_regex, variable):
                    if not isinstance(intervention_space[variable], dict):
                        intervention_dict[variable] = \
                            np.random.uniform(
                                intervention_space[variable][0],
                                intervention_space[variable][1])
                    else:
                        intervention_dict[variable] = dict()
                        for subvariable in intervention_space[variable]:
                            if re.fullmatch(self._second_level_regex, subvariable):
                                intervention_dict[variable][subvariable] = \
                                    np.random.uniform(
                                        intervention_space[variable][subvariable][0],
                                        intervention_space[variable][subvariable][1])
            return intervention_dict
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
