from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class SameMassesOOD(Protocol):
    def __init__(self):
        self.name = 'same_masses_ood'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task.testing_intervention_spaces
            mass = None
            for rigid_object in self.env._task.stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'mass' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    if mass is None:
                        mass = np.random.uniform(intervention_space[rigid_object]['mass'][0],
                                                 intervention_space[rigid_object]['mass'][1])
                    intervention_dict[rigid_object]['mass'] = mass
            return intervention_dict
        else:
            return None
