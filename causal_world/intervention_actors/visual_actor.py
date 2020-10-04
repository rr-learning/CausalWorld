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
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

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
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {'visual_actor': dict()}
