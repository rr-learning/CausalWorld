from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class InitialPosesOOD(Protocol):
    def __init__(self):
        super().__init__('initial_poses_ood')

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task._testing_intervention_spaces
            for rigid_object in self.env._task._stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'cartesian_position' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    intervention_dict[rigid_object]['cartesian_position'] = \
                        np.random.uniform(intervention_space[rigid_object]['cartesian_position'][0],
                                          intervention_space[rigid_object]['cartesian_position'][1])
            return intervention_dict
        else:
            return None
