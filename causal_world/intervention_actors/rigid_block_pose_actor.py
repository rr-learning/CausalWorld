import numpy as np
from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy


class RigidPoseInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, positions=True, orientations=True, **kwargs):
        """
        This intervention actor intervenes on the pose of the blocks
        available in the arena.

        :param positions: (bool) True if interventions on positions should be
                                 allowed.
        :param orientations: (bool) True if interventions on orientations should
                                    be allowed.
        :param kwargs:
        """
        super(RigidPoseInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None
        self.positions = positions
        self.orientations = orientations

    def initialize(self, env):
        """
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

        :return:
        """
        self.task_intervention_space = env.get_variable_space_used()
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if variable.startswith('tool'):
                interventions_dict[variable] = dict()
                if self.positions:
                    interventions_dict[variable]['cylindrical_position'] = \
                        np.random.uniform(
                            self.task_intervention_space
                            [variable]['cylindrical_position'][0],
                            self.task_intervention_space
                            [variable]['cylindrical_position'][1])
                if self.orientations:
                    interventions_dict[variable]['euler_orientation'] = \
                        np.random.uniform(
                            self.task_intervention_space
                            [variable]['euler_orientation'][0],
                            self.task_intervention_space
                            [variable]['euler_orientation'][1])
        return interventions_dict

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {
            'rigid_pose_actor': {
                'positions': self.positions,
                'orientations': self.orientations
            }
        }
