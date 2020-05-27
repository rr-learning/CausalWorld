import gym


class InterventionsCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, meta_actor_policy,
                 meta_episode_hold=1):
        super(InterventionsCurriculumWrapper, self).__init__(env)
        self.meta_actor_policy = meta_actor_policy
        self.meta_episode_hold = meta_episode_hold
        self._elapsed_steps = 0

    def reset(self):
        self._elapsed_steps += 1
        current_interventions_dict = dict()
        if self.meta_actor_policy and \
                self._elapsed_steps % self.meta_episode_hold == 0:
            current_interventions_dict.update(
                self.meta_actor_policy.act(
                    self.env.get_current_task_parameters()))
        else:
            current_interventions_dict = None
        obs = self.env.reset(interventions_dict=current_interventions_dict)
        return obs
