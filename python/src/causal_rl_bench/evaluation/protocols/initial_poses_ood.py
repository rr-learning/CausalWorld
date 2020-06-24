from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class InitialPosesOOD(Protocol):
    def __init__(self):
        self.name = 'initial_poses_ood'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task.testing_intervention_spaces
            for rigid_object in self.env._task.stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'position' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['position'] = \
                        np.random.uniform(intervention_space[rigid_object]['position'][0],
                                          intervention_space[rigid_object]['position'][1])
            return intervention_dict
        else:
            return None
