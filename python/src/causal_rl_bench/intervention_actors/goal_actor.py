from causal_rl_bench.intervention_actors.base_actor import \
    BaseInterventionActorPolicy


class GoalInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, **kwargs):
        """
        This class indicates the goal intervention actor, which an
        intervention actor that intervenes by sampling a new goal.

        :param kwargs:
        """
        super(GoalInterventionActorPolicy, self).__init__()
        self.goal_sampler_function = None

    def initialize(self, env):
        """

        :param env:
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

        :return:
        """
        return {'goal_actor': dict()}
