from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy
import numpy as np


class JointsInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, **kwargs):
        super(JointsInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize(self, env):
        if env.is_in_training_mode():
            self.task_intervention_space = \
                env.task.get_testing_intervention_spaces()
        else:
            self.task_intervention_space = \
                env.task.get_training_intervention_spaces()
        return

    def _act(self, variables_dict):
        interventions_dict = dict()
        interventions_dict['joint_positions'] = \
            np.random.uniform(variables_dict['joint_positions'][0],
                              variables_dict['joint_positions'][1])
        return interventions_dict

    def get_params(self):
        return {'joints_agent': dict()}
