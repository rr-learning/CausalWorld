class Curriculum(object):

    def __init__(self, intervention_actors, actives):
        """
        This corresponds to a curriculum object where it takes in
        the intervention actor and when are they supposed to be activated.

        :param intervention_actors: (list) list of intervention actors
        :param actives: (list of tuples) each tuple indicates (episode_start,
                                         episode_end, episode_periodicity,
                                         time_step_for_intervention)
        """
        self.intervention_actors = intervention_actors
        self.actives = actives

    def get_interventions(self, current_task_params, episode, time_step):
        """

        :param current_task_params:
        :param episode:
        :param time_step:

        :return:
        """
        interventions_dict = dict()
        for actor_index, active in enumerate(self.actives):
            in_episode = active[0] <= episode <= active[1]
            episode_hold = (episode - active[0]) % active[2] == 0
            time_step_hold = time_step == active[3]
            if in_episode and episode_hold and time_step_hold:
                interventions_dict.update(
                    self.intervention_actors[actor_index].act(
                        current_task_params))
        if len(interventions_dict) == 0:
            interventions_dict = None
        return interventions_dict

    def initialize_actors(self, env):
        """

        :param env:

        :return:
        """
        for intervention_actor in self.intervention_actors:
            intervention_actor.initialize(env)
        return

    def get_params(self):
        """

        :return:
        """
        params = dict()
        params['actor_params'] = dict()
        for actor in self.intervention_actors:
            params['actor_params'].update(actor.get_params())
        params['actives'] = self.actives
        return params
