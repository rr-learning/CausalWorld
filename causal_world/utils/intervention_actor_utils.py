from causal_world.intervention_actors import RigidPoseInterventionActorPolicy, \
    RandomInterventionActorPolicy, GoalInterventionActorPolicy,\
    JointsInterventionActorPolicy, PhysicalPropertiesInterventionActorPolicy, \
    VisualInterventionActorPolicy


def initialize_intervention_actors(actors_params):
    """

    :param actors_params:
    :return:
    """
    intervention_actors_list = []
    for actor_param in actors_params:
        if actor_param == 'random_actor':
            intervention_actors_list.append(
                RandomInterventionActorPolicy(**actors_params[actor_param]))
        elif actor_param == 'goal_actor':
            intervention_actors_list.append(
                GoalInterventionActorPolicy(**actors_params[actor_param]))
        elif actor_param == 'joints_actor':
            intervention_actors_list.append(
                JointsInterventionActorPolicy(**actors_params[actor_param]))
        elif actor_param == 'physical_properties_actor':
            intervention_actors_list.append(
                PhysicalPropertiesInterventionActorPolicy(
                    **actors_params[actor_param]))
        elif actor_param == 'rigid_pose_actor':
            intervention_actors_list.append(
                RigidPoseInterventionActorPolicy(**actors_params[actor_param]))
        elif actor_param == 'visual_actor':
            intervention_actors_list.append(
                VisualInterventionActorPolicy(**actors_params[actor_param]))
        else:
            raise Exception(
                "The intervention actor {} can't be loaded".format(actor_param))
    return intervention_actors_list
