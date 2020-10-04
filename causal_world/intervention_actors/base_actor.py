class BaseInterventionActorPolicy(object):
    def __init__(self, **kwargs):
        """
        This class indicates the interface of an intervention actor

        :param kwargs: (params) parameters for the construction of the actor.
        """
        return

    def initialize(self, env):
        """
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

        :return:
        """
        return

    def act(self, variables_dict):
        """
        This functions enables the intervention actor to decide on specific
        interventions.

        :param variables_dict: (dict) The current dict of variables that it
                                      can intervene on with their current
                                      values. (this can be a two level dict)

        :return: (dict) interventions decided by the intervention actor to
                        intervene on.
        """
        interventions_dict = self._act(variables_dict)
        self.__validate_intervention_dict(variables_dict, interventions_dict)
        return interventions_dict

    def _act(self, variables_dict):
        """

        :param variables_dict:

        :return:
        """
        return {}

    def __validate_intervention_dict(self, variables_dict, intervention_dict):
        """

        :param variables_dict:
        :param intervention_dict:

        :return:
        """
        for intervention in intervention_dict:
            if intervention not in variables_dict:
                raise Exception("the meta actor "
                                "performed an invalid intervention "
                                "on a variable that is not part of its input")

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        raise Exception("get params is not implemented")
