from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
import numpy as np


class VisualInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, **kwargs):
        """
        This intervention actor intervenes on all visual components of the
        robot, (i.e: colors).

        :param kwargs:
        """
        super(VisualInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize(self, env):
        """

        :param env:
        :return:
        """
        if env.is_in_training_mode():
            self.task_intervention_space =\
                env.get_task().get_training_intervention_spaces()
        else:
            self.task_intervention_space = \
                env.get_task().get_testing_intervention_spaces()
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:

        :return:
        """
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                if 'color' in self.task_intervention_space[variable]:
                    interventions_dict[variable] = dict()
                    interventions_dict[variable]['color'] = np.random.uniform(
                        self.task_intervention_space[variable]['color'][0],
                        self.task_intervention_space[variable]['color'][1])
            elif 'color' in variable:
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def get_params(self):
        """

        :return:
        """
        return {'visual_actor': dict()}
