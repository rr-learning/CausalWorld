from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class SameColorsOOD(Protocol):
    def __init__(self):
        super().__init__('same_colors_ood')

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task.testing_intervention_spaces
            color = None
            for rigid_object in self.env._task.stage._rigid_objects:
                if rigid_object in intervention_space and \
                        'color' in intervention_space[rigid_object]:
                    intervention_dict[rigid_object] = dict()
                    if color is None:
                        color = np.random.uniform(intervention_space[rigid_object]['color'][0],
                                                  intervention_space[rigid_object]['color'][1])
                    intervention_dict[rigid_object]['color'] = color
            return intervention_dict
        else:
            return None
