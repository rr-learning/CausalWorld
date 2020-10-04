class ProtocolBase(object):
    """
    Base Protocol from which each EvaluationProtocol inherits. Default number
    of evaluation protocols is 200
    :param name: (str) name of the protocol
    """
    def __init__(self, name):
        self.name = name
        self.num_evaluation_episodes_default = 200
        self.num_evaluation_episodes = self.num_evaluation_episodes_default

    def init_protocol(self, env, tracker, fraction=1):
        """
        Initializes protocol

        :param env: (CausalWorld) environment
        :param tracker: (Tracker)
        :param fraction: (float) fraction of episodes to be evaluated using
                                  the protocol (can be higher than one)

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
        """
        Used by the protocols to initialize some variables further after the
        environment is passed..etc.
        :return:
        """
        return

    def get_name(self):
        """
        Returns the name of the protocol

        :return: (str) protocol name
        """
        return self.name

    def get_num_episodes(self):
        """
        Returns the name of the evaluation episodes in this protocol

        :return: (int) number of episodes in protocol
        """
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        """
        Returns the interventions that are applied at a given timestep of the
        episode.

        :param episode: (int) episode number of the protocol
        :param timestep: (int) time step within episode
        :return: (dict) intervention dictionary
        """
        raise NotImplementedError()
