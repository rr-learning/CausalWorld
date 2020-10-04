import numpy as np


class CrossEntropyMethod(object):

    def __init__(self,
                 planning_horizon,
                 max_iterations,
                 population_size,
                 num_elite,
                 action_upper_bound,
                 action_lower_bound,
                 model,
                 epsilon=0.001,
                 alpha=0.25):
        """
        Cross entropy method optimizer to be used.

        :param planning_horizon: (int) horizon for planning.
        :param max_iterations: (int) number of iterations for CEM.
        :param population_size: (int) population size per iteration.
        :param num_elite: (int) number of elites per iteration.
        :param action_upper_bound: (nd.array) action upper bound to sample.
        :param action_lower_bound: (nd.array) action lower bound to sample.
        :param model: (causal_world.dynamics_model.SimulatorModel) model to
                                                                   be used.
        :param epsilon: (float) epsilon to stop iterating when reached.
        :param alpha: (alpha) alpha to be used when moving the unimodal
                              gaussian.
        """
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.num_elite = num_elite
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.planning_horizon = planning_horizon
        self.actions_mean = \
            (self.action_upper_bound + self.action_lower_bound)/2
        self.actions_mean = np.tile(np.expand_dims(self.actions_mean, 0),
                                    [self.planning_horizon, 1])
        self.actions_variance = np.square(self.action_upper_bound -
                                          self.action_lower_bound) / 16
        self.actions_variance = np.tile(
            np.expand_dims(self.actions_variance, 0),
            [self.planning_horizon, 1])

    def get_actions(self):
        """

        :return: (nd.array) getting the best actions after performing
                            CEM optimization.
        """
        best_reward = -np.float('inf')
        best_action = None
        iteration_index = 0
        current_actions_mean = np.array(self.actions_mean)
        current_actions_var = np.array(self.actions_variance)
        while iteration_index < self.max_iterations:
            print("cem iteration number: ", iteration_index)
            #TODO: change to truncated one
            action_samples = np.random.normal(
                current_actions_mean,
                np.sqrt(current_actions_var),
                size=[self.population_size, *self.actions_mean.shape])
            rewards = self.model.evaluate_trajectories(action_samples)
            elites_indicies = rewards.argsort(axis=0)[-self.num_elite:][::-1]
            best_current_reward = np.max(rewards)
            if best_current_reward > best_reward:
                best_reward = best_current_reward
                best_action = action_samples[np.argmax(rewards)]
            print("iteration's best reward is ", best_current_reward)
            elites = action_samples[elites_indicies]
            new_mean = np.mean(elites, axis=0)
            new_variance = np.var(elites, axis=0)
            current_actions_mean = (self.alpha * current_actions_mean) + (
                (1 - self.alpha) * new_mean)
            current_actions_var = (self.alpha * current_actions_var) + (
                (1 - self.alpha) * new_variance)
            iteration_index += 1
        return best_action
