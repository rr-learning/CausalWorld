import gym
from causal_world.curriculum import Curriculum


class CurriculumWrapper(gym.Wrapper):

    def __init__(self, env, intervention_actors, actives):
        """

        :param env:
        :param intervention_actors:
        :param actives:
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

        :param action:
        :return:
        """
        step_dict = self.env.step(action)
        self._elapsed_timesteps += 1
        interventions_dict = \
            self.interventions_curriculum.get_interventions(
                current_task_params=self.env.get_current_state_variables(),
                episode=self._elapsed_episodes,
                time_step=self._elapsed_timesteps)
        # perform intervention
        if interventions_dict is not None:
            intervention_success_signal = \
                self.env.do_intervention(interventions_dict=
                                         interventions_dict)
        # TODO: discuss that the observations now doesnot correspond to
        #  what u actually perofrmed?
        return step_dict

    def reset(self):
        """

        :return:
        """
        self._elapsed_episodes += 1
        interventions_dict = \
            self.interventions_curriculum.get_interventions(
                current_task_params=self.env.get_current_state_variables(),
                episode=self._elapsed_episodes,
                time_step=0)
        if interventions_dict is not None:
            success_signal, obs = self.env.set_starting_state(interventions_dict)
        else:
            obs = self.env.reset()
        return obs
