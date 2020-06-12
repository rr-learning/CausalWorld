from causal_rl_bench.evaluation.protocol import Protocol


class GoalPosesOOD(Protocol):
    def __init__(self):
        self.name = 'goal_poses_ood'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        if timestep == 0:
            return self.env.sample_new_goal(training=False, level=None)
        else:
            return None
