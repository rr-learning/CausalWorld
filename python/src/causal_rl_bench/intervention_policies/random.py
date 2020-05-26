from causal_rl_bench.intervention_policies.base_policy import BaseInterventionPolicy
import numpy as np


class StudentRandomInterventionPolicy(BaseInterventionPolicy):
    def __init__(self, task_intervention_space):
        super(StudentRandomInterventionPolicy, self).__init__()
        self.task_intervention_space = task_intervention_space
        self.sampler = None

    def act(self, variables_dict):
        interventions_dict = dict()
        for variable in variables_dict:
            if variable == "joint_positions":
                interventions_dict[variable] = self.sampler()
            else:
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def initialize_sampler(self, sampler_func):
        self.sampler = sampler_func


class TeacherRandomInterventionPolicy(BaseInterventionPolicy):
    def __init__(self, task_intervention_space):
        super(TeacherRandomInterventionPolicy, self).__init__()
        self.task_intervention_space = task_intervention_space
        self.sampler = None

    def act(self, variables_dict):
        interventions_dict = dict()
        for variable in variables_dict:
            #TODO: check for feasibility
            if variable == "goal_positions":
                interventions_dict[variable] = self.sampler()
            else:
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def initialize_sampler(self, sampler_func):
        self.sampler = sampler_func

