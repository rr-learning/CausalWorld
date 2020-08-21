from causal_world.evaluation.protocol import Protocol
import numpy as np


class Protocol10(Protocol):

    def __init__(self):
        """

        """
        super().__init__('P10')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            task_intervention_space = \
                self.env.get_task().get_intervention_space_a()
            interventions_dict = dict()
            for variable in task_intervention_space:
                if variable != 'joint_positions':
                    if isinstance(task_intervention_space[variable], dict):
                        interventions_dict[variable] = dict()
                        for subvariable_name in task_intervention_space[variable]:
                            interventions_dict[variable][subvariable_name] = \
                                np.random.uniform(
                                    task_intervention_space
                                    [variable][subvariable_name][0],
                                    task_intervention_space
                                    [variable][subvariable_name][1])
                    else:
                        interventions_dict[variable] = np.random.uniform(
                            task_intervention_space[variable][0],
                            task_intervention_space[variable][1])
            return interventions_dict
        else:
            return None
