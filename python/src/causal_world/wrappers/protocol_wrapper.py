import gym


class ProtocolWrapper(gym.Wrapper):

    def __init__(self, env, protocol):
        """

        :param env:
        :param protocol:
        """
        super(ProtocolWrapper, self).__init__(env)
        self.protocol = protocol
        self.env.add_wrapper_info(
            {'evaluation_environment': self.protocol.get_name()})
        self._elapsed_episodes = 0
        self._elapsed_timesteps = 0
        return

    def step(self, action):
        """

        :param action:
        :return:
        """
        observation, reward, done, info = self.env.step(action)
        self._elapsed_timesteps += 1
        invalid_interventions = 0
        interventions_dict = self.protocol.get_intervention(
            episode=self._elapsed_episodes, timestep=self._elapsed_episodes)
        if interventions_dict is not None:
            success_signal, observation = \
                self.env.do_intervention(interventions_dict=interventions_dict)
            while not success_signal and invalid_interventions < 5:
                invalid_interventions += 1
                interventions_dict = self.protocol.get_intervention(
                    episode=self._elapsed_episodes,
                    timestep=self._elapsed_episodes)
                if interventions_dict is not None:
                    success_signal, observation = \
                        self.env.do_intervention(interventions_dict=
                                                 interventions_dict)
                else:
                    break
        return observation, reward, done, info

    def reset(self):
        """

        :return:
        """
        self._elapsed_episodes += 1
        self._elapsed_timesteps = 0
        invalid_interventions = 0
        observation = self.env.reset()
        interventions_dict = self.protocol.get_intervention(
            episode=self._elapsed_episodes, timestep=0)
        if interventions_dict is not None:
            success_signal, observation = self.env.do_intervention(interventions_dict)
            while not success_signal and invalid_interventions < 5:
                invalid_interventions += 1
                interventions_dict = self.protocol.get_intervention(
                    episode=self._elapsed_episodes, timestep=0)
                if interventions_dict is not None:
                    success_signal, observation = self.env.do_intervention(
                        interventions_dict)
                else:
                    break
        return observation
