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

        :param current_task_params: (dict) specifies the current variables in
                                           the world and their values, its max
                                           2 levels dictionary for now.
        :param episode: (int) specifies the current episode number.
        :param time_step: (int) specifies the current time step index within
                                the episode.

        :return: (dict) returns a dictionary of all the variables decided to
                        intervene on by the actors.
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
        This function is used to initialize the actors. Basically it gives
        the intervention actors a chance to access the env and query about
        things like action space and so on.

        :param env: (causal_world.CausalWorld) The env used.

        :return:
        """
        for intervention_actor in self.intervention_actors:
            intervention_actor.initialize(env)
        return

    def get_params(self):
        """
        :return: (dict) returns the current status of the curriculum itself.
                        The actors used and so on.
        """
        params = dict()
        params['actor_params'] = dict()
        for actor in self.intervention_actors:
            params['actor_params'].update(actor.get_params())
        params['actives'] = self.actives
        return params
