import numpy as np
from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy


class RigidPoseInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self):
        super(RigidPoseInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize_actor(self, env):
        self.task_intervention_space =\
            env.task.get_testing_intervention_spaces()
        self.task_intervention_space.\
            update(env.task.get_training_intervention_spaces())
        return

    def _act(self, variables_dict):
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict) \
                    and variable.startswith("tool"):
                interventions_dict[variable] = dict()
                interventions_dict[variable]['position'] = \
                    np.random.uniform(
                        self.task_intervention_space
                        [variable]['position'][0],
                        self.task_intervention_space
                        [variable]['position'][1])
        return interventions_dict

    def get_intervention_actor_params(self):
        #TODO: We need to think about how to save its params more and load them?
        #potentially?
        intervention_params = dict()
        intervention_params["intervention_actor_name"] = \
            "rigid_block_position"
        return intervention_params
