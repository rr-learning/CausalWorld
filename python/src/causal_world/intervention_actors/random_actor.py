from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
import numpy as np


class RandomInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, **kwargs):
        """
        This is a random intervention actor which intervenes randomly on
        all available state variables.

        :param kwargs:
        """
        super(RandomInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize(self, env):
        """

        :param env:
        :return:
        """
        self.task_intervention_space = env.get_variable_space_used()
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                interventions_dict[variable] = dict()
                for subvariable_name in self.task_intervention_space[variable]:
                    interventions_dict[variable][subvariable_name] =\
                        np.random.uniform(
                        self.task_intervention_space
                        [variable][subvariable_name][0],
                        self.task_intervention_space
                        [variable][subvariable_name][1])
            else:
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def get_params(self):
        """

        :return:
        """
        return {'random_actor': dict()}
