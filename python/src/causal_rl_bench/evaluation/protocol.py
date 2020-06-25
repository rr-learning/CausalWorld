class Protocol(object):
    def __init__(self, name):
        self.name = name
        self.num_evaluation_episodes = 50

    def init_protocol(self, env, tracker):
        self.env = env
        self.tracker = tracker

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
        raise NotImplementedError()
