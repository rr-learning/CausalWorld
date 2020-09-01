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
        self._inverse_kinemetics_func = None
        self._stage_bb = None

    def initialize(self, env):
        """

        :param env:
        :return:
        """
        self.task_intervention_space = env.get_variable_space_used()
        self._inverse_kinemetics_func = env.get_robot().inverse_kinematics
        self._stage_bb = env.get_stage().get_stage_bb()
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        interventions_dict = dict()
        desired_tip_positions = np.random.uniform(self._stage_bb[0],
                                                  self._stage_bb[1],
                                                  size=[3, 3]).flatten()

        interventions_dict['joint_positions'] = \
            self._inverse_kinemetics_func(desired_tip_positions,
                                          rest_pose=np.zeros(9,).tolist())
        return interventions_dict

    def get_params(self):
        """
        
        :return:
        """
        return {'joints_actor': dict()}
