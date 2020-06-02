from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy


class GoalInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, **kwargs):
        super(GoalInterventionActorPolicy, self).__init__()
        self.goal_sampler_function = None

    def initialize(self, env):
        self.goal_sampler_function = env.sample_new_goal
        return

    def _act(self, variables_dict):
        return self.goal_sampler_function()

    def get_params(self):
        return {'goal_agent': dict()}
