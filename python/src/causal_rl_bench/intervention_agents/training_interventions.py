from causal_rl_bench.intervention_agents.reacher_randomizer \
    import ReacherInterventionActorPolicy
from causal_rl_bench.intervention_agents.rigid_block_pose \
    import RigidPoseInterventionActorPolicy


def get_reset_training_intervention_agent(task_generator_id="reaching",
                                          **kwargs):
    if task_generator_id == "reaching":
        training_intervention_agent = ReacherInterventionActorPolicy()
    elif task_generator_id == "pushing":
        training_intervention_agent = RigidPoseInterventionActorPolicy()
    elif task_generator_id == "picking":
        training_intervention_agent = RigidPoseInterventionActorPolicy()
    else:
        raise Exception("No default intervention agent for this task")
    return training_intervention_agent
