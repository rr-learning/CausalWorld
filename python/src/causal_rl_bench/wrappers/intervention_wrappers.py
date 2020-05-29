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
