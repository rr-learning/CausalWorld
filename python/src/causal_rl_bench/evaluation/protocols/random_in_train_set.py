from causal_rl_bench.evaluation.protocol import Protocol
import numpy as np


class RandomInTrainSet(Protocol):
    def __init__(self):
        self.name = 'random_in_train_set'
        self.num_evaluation_episodes = 10

    def get_name(self):
        return self.name

    def get_num_episodes(self):
        return self.num_evaluation_episodes

    def get_intervention(self, episode, timestep):
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
