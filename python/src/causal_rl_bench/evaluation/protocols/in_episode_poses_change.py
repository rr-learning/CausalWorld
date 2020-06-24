from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class InEpisodePosesChange(Protocol):
    def __init__(self):
        super().__init__('in_episode_poses_change')

    def get_intervention(self, episode, timestep):
        # Arbitrary choice for timestep here
        if timestep == 30:
            intervention_dict = dict()
            intervention_space = self.env._task.training_intervention_spaces
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
