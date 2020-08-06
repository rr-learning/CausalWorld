from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
import numpy as np


class JointsInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, **kwargs):
        """
        This class indicates the joint intervention actor which intervenes on
        the joints of the robot in a random fashion.

        :param kwargs:
        """
        super(JointsInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None

    def initialize(self, env):
        """

        :param env:
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
        interventions_dict['joint_positions'] = \
            np.random.uniform(self.task_intervention_space['joint_positions'][0],
                              self.task_intervention_space['joint_positions'][1])
        return interventions_dict

    def get_params(self):
        """
        
        :return:
        """
        return {'joints_actor': dict()}
