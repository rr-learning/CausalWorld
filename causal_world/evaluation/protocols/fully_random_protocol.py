from causal_world.evaluation.protocols.protocol import ProtocolBase
import numpy as np


class FullyRandomProtocol(ProtocolBase):

    def __init__(self, name, variable_space='space_a_b'):
        """
        This specifies a fully random protocol, where an intervention is
        produced on every exposed variable by uniformly sampling the
        intervention space.

        :param name: (str) specifies the name of the protocol to be reported.
        :param variable_space: (str) "space_a", "space_b" or "space_a_b".
        """
        super().__init__(name)
        self._variable_space = variable_space

    def get_intervention(self, episode, timestep):
        """
        Returns the interventions that are applied at a given timestep of the
        episode.

        :param episode: (int) episode number of the protocol
        :param timestep: (int) time step within episode
        :return: (dict) intervention dictionary
        """
        if timestep == 0:
            if self._variable_space == 'space_a_b':
                intervention_space = self.env.get_intervention_space_a_b()
            elif self._variable_space == 'space_a':
                intervention_space = self.env.get_intervention_space_a()
            elif self._variable_space == 'space_b':
                intervention_space = self.env.get_intervention_space_b()
            interventions_dict = dict()
            intervene_on_size = np.random.choice([0, 1], p=[0.5, 0.5])
            intervene_on_joint_positions = np.random.choice([0, 1],
                                                            p=[1, 0])
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
        """
        Used by the protocols to initialize some variables further after the
        environment is passed..etc.

        :return:
        """
        if self._variable_space == 'space_a_b':
            self.env.set_intervention_space(variables_space='space_a_b')
        elif self._variable_space == 'space_a':
            self.env.set_intervention_space(variables_space='space_a')
        elif self._variable_space == 'space_b':
            self.env.set_intervention_space(variables_space='space_b')
        return
