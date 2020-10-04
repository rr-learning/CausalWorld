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
        #choose randomly to intervene on size OR cylindrical position since
        #size affects cylindrical position
        intervene_on_size = np.random.choice([0, 1], p=[0.5, 0.5])
        intervene_on_joint_positions = np.random.choice([0, 1], p=[1, 0])
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
                interventions_dict[variable] = np.random.uniform(
                    self.task_intervention_space[variable][0],
                    self.task_intervention_space[variable][1])
        return interventions_dict

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {'random_actor': dict()}
