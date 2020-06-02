import gym


class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, interventions_curriculum):
        super(CurriculumWrapper, self).__init__(env)
        self.interventions_curriculum = interventions_curriculum
        self.interventions_curriculum.initialize_actors(env=env)
        self.interventions_curriculum.reset()
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


class ActionsInterventionsActorWrapper(gym.Wrapper):
    def __init__(self, env, intervention_actor):
        super(ActionsInterventionsActorWrapper, self).__init__(env)
        self.env = env
        self.intervention_actor = intervention_actor
        self.intervention_actor.initialize_actor(self.env)
        self.env.disable_actions()

    def step(self, action):
        intervention_dict = self.intervention_actor.act(
            self.env.get_current_task_parameters())
        self.env.do_intervention(intervention_dict)
        return self.env.step(self.env.action_space.low)

    def get_world_params(self):
        #TODO: propagate this to other wrappers
        default_world_params = self.env.get_world_params()
        default_world_params.update(
            self.intervention_actor.get_intervention_actor_params())
        return default_world_params
