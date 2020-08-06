from causal_world.evaluation.protocol import Protocol


class GoalPosesSpaceA(Protocol):

    def __init__(self):
        """

        """
        super().__init__('goal_poses_space_A')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """

        if timestep == 0:
            return self.env.sample_new_goal(level=None)
        else:
            return None

    def _init_protocol_helper(self):
        self.env.set_intervention_space(variables_space='space_a')
        return
