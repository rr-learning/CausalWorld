class Protocol(object):
    def get_name(self):
        raise NotImplementedError()

    def get_num_episodes(self):
        raise NotImplementedError()

    def get_intervention(self, env, episode, timestep):
        raise NotImplementedError()
