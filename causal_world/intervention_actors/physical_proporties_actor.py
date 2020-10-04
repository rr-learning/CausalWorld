from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
import numpy as np


class PhysicalPropertiesInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, group, **kwargs):
        """
        This intervention actor intervenes on physcial proporties such as
        friction, mass...etc

        :param group: (str) the object that the actor will intervene on.
                            floor, stage, robot..etc
        :param kwargs:
        """
        #group can be robot stage floor or tool
        super(PhysicalPropertiesInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None
        self.group = group

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
            if variable.startswith(self.group):
                if isinstance(self.task_intervention_space[variable], dict):
                    if 'mass' in self.task_intervention_space[variable]:
                        interventions_dict[variable] = dict()
                        interventions_dict[variable]['mass'] = np.random.uniform(
                            self.task_intervention_space[variable]['mass'][0],
                            self.task_intervention_space[variable]['mass'][1])
                    elif 'friction' in self.task_intervention_space[variable]:
                        interventions_dict[variable] = dict()
                        interventions_dict[variable][
                            'friction'] = np.random.uniform(
                                self.task_intervention_space[variable]
                                ['friction'][0],
                                self.task_intervention_space[variable]
                                ['friction'][1])
                elif 'mass' in variable:
                    interventions_dict[variable] = np.random.uniform(
                        self.task_intervention_space[variable][0],
                        self.task_intervention_space[variable][1])
                elif 'friction' in variable:
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
        return {'physical_properties_actor': {'group': self.group}}
