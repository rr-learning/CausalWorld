class Curriculum(object):
    def __init__(self, intervention_actors, actives, **kwargs):
        self.intervention_actors = intervention_actors
        self.actives = actives

    def get_interventions(self, current_task_params, episode, time_step):
        interventions_dict = dict()
        for actor_index, active in enumerate(self.actives):
            in_episode = active[0] <= episode <= active[1]
            episode_hold = (episode - active[0]) % active[2] == 0
            time_step_hold = time_step == active[3]
            if in_episode and episode_hold and time_step_hold:
                interventions_dict.update(
                    self.intervention_actors[actor_index].act(current_task_params))
        if len(interventions_dict) == 0:
            interventions_dict = None
        return interventions_dict

    def initialize_actors(self, env):
        for intervention_actor in self.intervention_actors:
            intervention_actor.initialize(env)
        return

    def get_params(self):
        params = dict()
        params['agent_params'] = dict()
        for actor in self.intervention_actors:
            params['agent_params'].update(actor.get_params())
        return params
