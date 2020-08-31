from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
import numpy as np


class RandomInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, **kwargs):
        """
        This is a random intervention actor which intervenes randomly on
        all available state variables except joint positions since its a
        trickier space.

        :param kwargs:
        """
        super(RandomInterventionActorPolicy, self).__init__()
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
        self._stage_bb = env.get_stage()._get_stage_bb()
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        #choose randomly to intervene on size OR cylindrical position since
        #size affects cylindrical position
        intervene_on_size = np.random.choice([0, 1], p=[0.5, 0.5])
        intervene_on_joint_positions = np.random.choice([0, 1], p=[0.9, 0.1])
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                interventions_dict[variable] = dict()
                for subvariable_name in self.task_intervention_space[variable]:
                    if subvariable_name == 'cylindrical_position' and \
                            intervene_on_size:
                        continue
                    if subvariable_name == 'size' and not intervene_on_size:
                        continue
                    interventions_dict[variable][subvariable_name] =\
                        np.random.uniform(
                        self.task_intervention_space
                        [variable][subvariable_name][0],
                        self.task_intervention_space
                        [variable][subvariable_name][1])
            else:
                if not intervene_on_joint_positions and variable == 'joint_positions':
                    continue
                if intervene_on_joint_positions and variable == 'joint_positions':
                    desired_tip_positions = np.random.uniform(
                        self._stage_bb[0],
                        self._stage_bb[1],
                        size=[3, 3]).flatten()

                    interventions_dict[variable] = \
                        self._inverse_kinemetics_func(desired_tip_positions,
                                                      rest_pose=np.zeros(
                                                          9, ).tolist())
                    continue
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def get_params(self):
        """

        :return:
        """
        return {'random_actor': dict()}
