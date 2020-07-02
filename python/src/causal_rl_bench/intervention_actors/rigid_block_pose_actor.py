import numpy as np
from causal_rl_bench.intervention_actors.base_actor import \
    BaseInterventionActorPolicy


class RigidPoseInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, positions=True, orientations=True, **kwargs):
        """

        :param positions:
        :param orientations:
        :param kwargs:
        """
        super(RigidPoseInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None
        self.positions = positions
        self.orientations = orientations

    def initialize(self, env):
        """

        :param env:
        :return:
        """
        self.task_intervention_space =\
            env._task.get_testing_intervention_spaces()
        self.task_intervention_space.\
            update(env._task.get_training_intervention_spaces())
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
                    interventions_dict[variable]['orientation'] = \
                        np.random.uniform(
                            self.task_intervention_space
                            [variable]['orientation'][0],
                            self.task_intervention_space
                            [variable]['orientation'][1])
        return interventions_dict

    def get_params(self):
        """

        :return:
        """
        return {'rigid_pose_actor': {'positions': self.positions,
                                     'orientations': self.orientations}}
