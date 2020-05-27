from causal_rl_bench.meta_agents.base_policy import BaseMetaActorPolicy
import numpy as np


class RandomMetaActorPolicy(BaseMetaActorPolicy):
    def __init__(self, task_intervention_space):
        super(RandomMetaActorPolicy, self).__init__()
        self.task_intervention_space = task_intervention_space
        self.sampler_funcs = dict()

    def act(self, variables_dict):
        interventions_dict = dict()
        for variable in variables_dict:
            if isinstance(variables_dict[variable], dict):
                for subvariable_name in variables_dict[variable]:
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
