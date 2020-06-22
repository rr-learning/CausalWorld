import gym


class ProtocolWrapper(gym.Wrapper):
    def __init__(self, env, protocol):
        super(ProtocolWrapper, self).__init__(env)
        self.protocol = protocol
        self.env._add_wrapper_info({'evaluation_environment':
                                    self.protocol.get_name()})
        self._elapsed_episodes = 0
        self._elapsed_timesteps = 0
        return

    def step(self, action):
        step_dict = self.env.step(action)
        self._elapsed_timesteps += 1
        interventions_dict = self.protocol.get_intervention(episode=self._elapsed_episodes,
                                                            timestep=self._elapsed_episodes)
        if interventions_dict is not None:
            self.env.do_intervention(interventions_dict=interventions_dict)
        return step_dict

    def reset(self):
        self._elapsed_episodes += 1
        self._elapsed_timesteps = 0
        interventions_dict = self.protocol.get_intervention(episode=self._elapsed_episodes,
                                                            timestep=0)
        observation = self.env.reset()
        if interventions_dict is not None:
            self.env.do_intervention(interventions_dict)
            observation = self.env.task.filter_structured_observations()
        return observation
