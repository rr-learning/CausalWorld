from causal_world.evaluation.protocol import Protocol
import numpy as np


class RandomInTrainSet(Protocol):

    def __init__(self):
        """

        """
        super().__init__('random_in_train_set')

    def get_intervention(self, episode, timestep):
        """

        :param episode:
        :param timestep:

        :return:
        """
        if timestep == 0:
            task_intervention_space = \
                self.env._task.get_training_intervention_spaces()
            interventions_dict = dict()
            for variable in task_intervention_space:
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