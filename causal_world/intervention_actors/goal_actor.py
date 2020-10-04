from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy


class GoalInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, **kwargs):
        """
        This class indicates the goal intervention actor, which an
        intervention actor that intervenes by sampling a new goal.

        :param kwargs: (params) parameters for the construction of the actor.
        """
        super(GoalInterventionActorPolicy, self).__init__()
        self.goal_sampler_function = None

    def initialize(self, env):
        """
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

        :return:
        """
        self.goal_sampler_function = env.sample_new_goal
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        return self.goal_sampler_function()

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {'goal_actor': dict()}
