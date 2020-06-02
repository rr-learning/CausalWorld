from causal_rl_bench.intervention_agents import RigidPoseInterventionActorPolicy, \
    ReacherInterventionActorPolicy, RandomInterventionActorPolicy, GoalInterventionActorPolicy,\
    JointsInterventionActorPolicy, PhysicalPropertiesInterventionActorPolicy, VisualInterventionActorPolicy


def initialize_intervention_agents(agents_params):
    intervention_actors_list = []
    for agent_param in agents_params:
        if agent_param == 'random_agent':
            intervention_actors_list.append(RandomInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'goal_agent':
            intervention_actors_list.append(GoalInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'joints_agent':
            intervention_actors_list.append(JointsInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'physical_properties_agent':
            intervention_actors_list.append(PhysicalPropertiesInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'reacher_agent':
            intervention_actors_list.append(ReacherInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'rigid_pose_agent':
            intervention_actors_list.append(RigidPoseInterventionActorPolicy(**agents_params[agent_param]))
        elif agent_param == 'visual_agent':
            intervention_actors_list.append(VisualInterventionActorPolicy(**agents_params[agent_param]))
        else:
            raise Exception("The intervention agent {} can't be loaded".format(agent_param))
    return intervention_actors_list
