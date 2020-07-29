from causal_world.evaluation.protocol import Protocol


class GoalPosesTestSpace(Protocol):

    def __init__(self):
        """

        """
        super().__init__('goal_poses_test_space')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            return self.env.sample_new_goal(training=False, level=None)
        else:
            return None