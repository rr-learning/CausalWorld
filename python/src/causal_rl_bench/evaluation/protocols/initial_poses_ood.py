from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class InitialPosesOOD(Protocol):
    def __init__(self):
        super().__init__('initial_poses_ood')

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env.task.testing_intervention_spaces
            for rigid_object in self.env.task.stage.rigid_objects:
                if rigid_object in intervention_space and \
                        'position' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['position'] = \
                        np.random.uniform(intervention_space[rigid_object]['position'][0],
                                          intervention_space[rigid_object]['position'][1])
            return intervention_dict
        else:
            return None
