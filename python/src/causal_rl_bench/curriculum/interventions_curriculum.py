import gym


class InterventionsCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, student_policy=None,
                 student_episode_hold=1,
                 teacher_policy=None,
                 teacher_episode_hold=1,
                 master_policy=None,
                 master_episode_hold=1):
        super(InterventionsCurriculumWrapper, self).__init__(env)
        self.student_policy = student_policy
        self.student_episode_hold = student_episode_hold
        self._elapsed_steps = 0
        self.teacher_policy = teacher_policy
        self.teacher_episode_hold = teacher_episode_hold
        self.master_policy = master_policy
        self.master_episode_hold = master_episode_hold

    def reset(self):
        self._elapsed_steps += 1
        current_interventions_dict = dict()
        if self.student_policy and self._elapsed_steps % self.student_episode_hold == 0:
            current_interventions_dict.update(
                self.student_policy.act(self.env.get_current_student_params()))
        if self.teacher_policy and self._elapsed_steps % self.teacher_episode_hold == 0:
            current_interventions_dict.update(
                self.teacher_policy.act(self.env.get_current_teacher_params()))
        if self.master_policy and self._elapsed_steps % self.master_episode_hold == 0:
            current_interventions_dict.update(
                self.master_policy.act(self.env.get_current_master_params()))
        if len(current_interventions_dict) == 0:
            current_interventions_dict = None
        obs = self.env.reset(interventions_dict=current_interventions_dict)
        return obs
