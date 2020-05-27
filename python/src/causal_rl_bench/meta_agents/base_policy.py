"""
causal_rl_bench/meta_agents/base_policy.py
=========================================
"""


class BaseMetaActorPolicy(object):
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
        #this should return interventions dict on certain variables
        raise Exception("not implemented yet")
