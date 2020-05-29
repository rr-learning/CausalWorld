from causal_rl_bench.intervention_agents.reacher_randomizer \
    import ReacherInterventionActorPolicy


def reset_training_intervention_agent(task_generator_id="reaching",
                                      **kwargs):
    if task_generator_id == "reaching":
        training_intervention_agent = ReacherInterventionActorPolicy()
    else:
        raise Exception("No default intervention agent for this task")
    return training_intervention_agent
