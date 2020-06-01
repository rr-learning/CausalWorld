import gym


class InterventionsCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, interventions_curriculum):
        super(InterventionsCurriculumWrapper, self).__init__(env)
        self.intervention_actors = \
            interventions_curriculum.intervention_actors
        self.episode_holds = \
            interventions_curriculum.episode_holds
        self._elapsed_episodes = 0

    def reset(self):
        #at each reset we check which intervention actor will intervene if
        #any
        self._elapsed_episodes += 1
        current_interventions_dict = dict()
        for i in range(len(self.intervention_actors)):
            if self._elapsed_episodes % self.episode_holds[i] == 0:
                current_intervention_actor = self.intervention_actors[i]
                current_interventions_dict.update(
                    current_intervention_actor.act(
                        self.env.get_current_task_parameters()))
        if len(current_interventions_dict) == 0:
            current_interventions_dict = None
        obs = self.env.reset(interventions_dict=current_interventions_dict)
        return obs


class ResetInterventionsActorWrapper(gym.Wrapper):
    def __init__(self, env, intervention_actor):
        super(ResetInterventionsActorWrapper, self).__init__(env)
        self.env = env
        self.intervention_actor = intervention_actor
        self.intervention_actor.initialize_actor(self.env)

    def reset(self):
        intervention_dict = self.intervention_actor.act(
            self.env.get_current_task_parameters())
        if len(intervention_dict) == 0:
            current_interventions_dict = None
        obs = self.env.reset(interventions_dict=intervention_dict)
        return obs

    def get_world_params(self):
        #TODO: propagate this to other wrappers
        default_world_params = self.env.get_world_params()
        default_world_params.update(
            self.intervention_actor.get_intervention_actor_params())
        return default_world_params


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
