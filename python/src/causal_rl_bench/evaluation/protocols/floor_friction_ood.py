from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class FloorFrictionOOD(Protocol):

    def __init__(self):
        super().__init__('floor_friction_ood')

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            intervention_dict = dict()
            intervention_space = self.env._task._testing_intervention_spaces
            floor_friction = np.random.uniform(
                intervention_space['floor_friction'][0],
                intervention_space['floor_friction'][1])
            intervention_dict['floor_friction'] = floor_friction
            return intervention_dict
        else:
            return None
