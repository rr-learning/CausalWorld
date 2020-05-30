"""
causal_rl_bench/meta_agents/base_policy.py
=========================================
"""


class BaseInterventionActorPolicy(object):
    """This class indicates the interface of a meta actor"""
    def __init__(self):
        return

    def act(self, variables_dict):
        """
        This functions enables the meta actor to decide on specific
        interventions.
        Parameters
        ---------
            variables_dict: dict
                The current dict of variables that it can intervene on with
                their current values. (this can be a two level dict)
        Returns
        -------
            interventions_dict: dict
               Dict of variables that the meta actor decided to intervene on
               with the corresponding values.
        """
        interventions_dict = self._act(variables_dict)
        self.validate_intervention_dict(variables_dict, interventions_dict)
        return interventions_dict

    def _act(self, variables_dict):
        return {}

    def initialize_actor(self, env):
        return

    def validate_intervention_dict(self, variables_dict, intervention_dict):
        #TODO: remove redundant interventions here
        for intervention in intervention_dict:
            if intervention not in variables_dict:
                raise Exception("the meta actor "
                                "performed an invalid intervention "
                                "on a variable that is not part of its input")
