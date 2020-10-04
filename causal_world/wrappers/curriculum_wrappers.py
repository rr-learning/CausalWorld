import gym
from causal_world.curriculum import Curriculum


class CurriculumWrapper(gym.Wrapper):

    def __init__(self, env, intervention_actors, actives):
        """

        :param env: (causal_world.CausalWorld) the environment to convert.
        :param intervention_actors: (list) list of intervention actors
        :param actives: (list of tuples) each tuple indicates (episode_start,
                                         episode_end, episode_periodicity,
                                         time_step_for_intervention)
        """
        # TODO: this wrapper can't be loaded at the moment or saved
        super(CurriculumWrapper, self).__init__(env)
        self.interventions_curriculum = Curriculum(
            intervention_actors=intervention_actors, actives=actives)
        self.interventions_curriculum.initialize_actors(env=env)
        self.env.add_wrapper_info({
            'curriculum_environment':
                self.interventions_curriculum.get_params()
        })
        self._elapsed_episodes = -1
        self._elapsed_timesteps = 0
        return

    def step(self, action):
        """
        Used to step through the enviroment.

        :param action: (nd.array) specifies which action should be taken by
                                  the robot, should follow the same action
                                  mode specified.

        :return: (nd.array) specifies the observations returned after stepping
                            through the environment. Again, it follows the
                            observation_mode specified.
        """
        observation, reward, done, info = self.env.step(action)
        invalid_interventions = 0
        self._elapsed_timesteps += 1
        interventions_dict = \
            self.interventions_curriculum.get_interventions(
                current_task_params=self.env.get_current_state_variables(),
                episode=self._elapsed_episodes,
                time_step=self._elapsed_timesteps)
        if interventions_dict is not None:
            success_signal, observation = \
                self.env.do_intervention(interventions_dict=
                                         interventions_dict)
            while not success_signal and invalid_interventions < 5:
                invalid_interventions += 1
                interventions_dict = \
                    self.interventions_curriculum.get_interventions(
                        current_task_params=self.env.get_current_state_variables(),
                        episode=self._elapsed_episodes,
                        time_step=self._elapsed_timesteps)
                if interventions_dict is not None:
                    success_signal, observation = \
                        self.env.do_intervention(interventions_dict=
                                                 interventions_dict)
                else:
                    break
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment to the current starting state of the environment.

        :return: (nd.array) specifies the observations returned after resetting
                            the environment. Again, it follows the
                            observation_mode specified.
        """
        self._elapsed_episodes += 1
        invalid_interventions = 0
        interventions_dict = \
            self.interventions_curriculum.get_interventions(
                current_task_params=self.env.get_current_state_variables(),
                episode=self._elapsed_episodes,
                time_step=0)
        if interventions_dict is not None:
            success_signal, obs = self.env.set_starting_state(interventions_dict)
            while not success_signal and invalid_interventions < 5:
                invalid_interventions += 1
                interventions_dict = \
                    self.interventions_curriculum.get_interventions(
                        current_task_params=self.env.get_current_state_variables(),
                        episode=self._elapsed_episodes,
                        time_step=0)
                if interventions_dict is not None:
                    success_signal, obs = self.env.set_starting_state(
                        interventions_dict)
                else:
                    obs = self.env.reset()
                    break

        else:
            obs = self.env.reset()
        return obs
