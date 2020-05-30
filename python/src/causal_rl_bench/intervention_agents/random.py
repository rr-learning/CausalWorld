from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy
import numpy as np


class RandomInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self):
        super(RandomInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None
        self.sampler_funcs = dict()

    def initialize_actor(self, env):
        self.task_intervention_space =\
            env.task.get_testing_intervention_spaces()
        return

    def _act(self, variables_dict):
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                interventions_dict[variable] = dict()
                for subvariable_name in self.task_intervention_space[variable]:
                    if variable in self.sampler_funcs and \
                            subvariable_name in self.sampler_funcs[variable]:
                        interventions_dict[variable][subvariable_name] = \
                            self.sampler_funcs[variable][subvariable_name]()
                    else:
                        interventions_dict[variable][subvariable_name] =\
                            np.random.uniform(
                            self.task_intervention_space
                            [variable][subvariable_name][0],
                            self.task_intervention_space
                            [variable][subvariable_name][1])
            else:
                if variable in self.sampler_funcs:
                    interventions_dict[variable] = \
                        self.sampler_funcs[variable]()
                else:
                    interventions_dict[variable] = np.random.uniform(
                        self.task_intervention_space[variable][0],
                        self.task_intervention_space[variable][1])
        return interventions_dict

    def add_sampler_func(self, variable_name, sampler_func,
                         subvariable_name=None):
        if subvariable_name:
            self.sampler_funcs[variable_name] = dict()
            self.sampler_funcs[variable_name][subvariable_name] = sampler_func
        else:
            self.sampler_funcs[variable_name] = sampler_func
        return
