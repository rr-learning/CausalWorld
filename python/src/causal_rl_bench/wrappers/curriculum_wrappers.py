import gym


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, interventions_curriculum):
        #TODO: this wrapper can't be loaded at the moment or saved
        super(CurriculumWrapper, self).__init__(env)
        self.interventions_curriculum = interventions_curriculum
        self.interventions_curriculum.initialize_actors(env=env)
        self.interventions_curriculum.reset()
        self.env.add_wrapper_info({'curriculum_environment': interventions_curriculum.get_params()})
        return

    def step(self, action):
        interventions_dict = \
            self.interventions_curriculum.get_in_episode_interventions(self.env.get_current_task_parameters())
        #perform intervention
        if interventions_dict is not None:
            intervention_success_signal = \
                self.env.do_intervention(interventions_dict=interventions_dict)
        #TODO: discuss that the observations now doesnot correspond to what u actually perofrmed?
        return self.env.step(action)

    def reset(self):
        interventions_dict = \
            self.interventions_curriculum.get_new_episode_interventions(self.env.get_current_task_parameters())
        return self.env.reset(interventions_dict)
