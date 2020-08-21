class Protocol(object):

    def __init__(self, name):
        """

        :param name:
        """
        self.name = name
        self.num_evaluation_episodes_default = 200
        self.num_evaluation_episodes = self.num_evaluation_episodes_default

    def init_protocol(self, env, tracker, fraction=1):
        """

        :param env:
        :param tracker:
        :param fraction:

        :return:
        """
        self.env = env
        self.env.set_intervention_space(variables_space='space_a_b')
        self.tracker = tracker
        if fraction > 0:
            self.num_evaluation_episodes = int(
                self.num_evaluation_episodes_default * fraction)

        else:
            raise ValueError(
                'fraction of episodes for evaluation needs to be strictly positive'
            )

        self._init_protocol_helper()
        return

    def _init_protocol_helper(self):
        return

    def get_name(self):
        """

        :return:
        """
        return self.name

    def get_num_episodes(self):
        """

        :return:
        """
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        raise NotImplementedError()
