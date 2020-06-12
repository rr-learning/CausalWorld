from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class ObjectSizesOOD(Protocol):
    def __init__(self):
        self.name = 'object_sizes_ood'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.task.testing_intervention_spaces
            for rigid_object in self.env.task.stage.rigid_objects:
                if rigid_object in intervention_space and \
                        'size' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['size'] = \
                        np.random.uniform(intervention_space[rigid_object]['size'][0],
                                          intervention_space[rigid_object]['size'][1])
            return intervention_dict
        else:
            return None
