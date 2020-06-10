class Protocol(object):
    def init(self, env, tracker):
        self.env = env
        self.tracker = tracker

    def get_name(self):
        raise NotImplementedError()

    def get_num_episodes(self):
        raise NotImplementedError()

    def get_intervention(self, episode, timestep):
        raise NotImplementedError()
