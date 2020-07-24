from causal_rl_bench.evaluation.protocol import Protocol


class GoalPosesOOD(Protocol):

    def __init__(self):
        super().__init__('goal_poses_ood')

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            return self.env.sample_new_goal(training=False, level=None)
        else:
            return None
