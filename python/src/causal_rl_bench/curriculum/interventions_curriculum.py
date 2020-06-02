class InterventionsCurriculum(object):
    def __init__(self, intervention_actors, episodes_hold, timesteps_hold):
        self.intervention_actors = intervention_actors
        self.episode_holds = episodes_hold
        self.timesteps_hold = timesteps_hold
        self._elapsed_episodes = 0
        self._elapsed_timesteps = 0

    def reset(self):
        self._elapsed_episodes = 0
        self._elapsed_timesteps = 0
        return

    def get_new_episode_interventions(self, current_task_params):
        self._elapsed_episodes += 1
        self._elapsed_timesteps = 0
        current_interventions_dict = dict()
        for i in range(len(self.intervention_actors)):
            if self.episode_holds[i] is not None and \
                    self._elapsed_episodes % self.episode_holds[i] == 0:
                current_intervention_actor = self.intervention_actors[i]
                current_interventions_dict.update(
                    current_intervention_actor.act(current_task_params))
        if len(current_interventions_dict) == 0:
            current_interventions_dict = None
        return current_interventions_dict

    def get_in_episode_interventions(self, current_task_params):
        self._elapsed_timesteps += 1
        current_interventions_dict = dict()
        for i in range(len(self.intervention_actors)):
            if self.timesteps_hold[i] is not None\
                    and self._elapsed_timesteps % self.timesteps_hold[i] == 0:
                current_intervention_actor = self.intervention_actors[i]
                current_interventions_dict.update(
                    current_intervention_actor.act(current_task_params))
        if len(current_interventions_dict) == 0:
            current_interventions_dict = None
        return current_interventions_dict

    def initialize_actors(self, env):
        for intervention_actor in self.intervention_actors:
            intervention_actor.initialize(env)
        return
