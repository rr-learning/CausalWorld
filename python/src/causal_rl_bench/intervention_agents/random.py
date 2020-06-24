from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy
import numpy as np


class RandomInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(RandomInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize(self, env):
        """

        :param env:
        :return:
        """
        if env.is_in_training_mode():
            self.task_intervention_space =\
                env._task.get_testing_intervention_spaces()
        else:
            self.task_intervention_space = \
                env._task.get_training_intervention_spaces()
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
        return {'random_agent': dict()}
