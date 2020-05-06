import numpy as np


class CrossEntropyMethod(object):
    def __init__(self, planning_horizon, max_iterations, population_size,
                 num_elite, action_upper_bound, action_lower_bound,
                 model, num_agents, epsilon=0.001, alpha=0.25):
        #TODO: support a learned model too
        self.num_agents = num_agents
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
        self.actions_mean = np.tile(np.expand_dims(np.expand_dims(self.actions_mean, 0), 0),
                                    [self.planning_horizon, self.num_agents, 1])
        self.actions_variance = np.square(
            self.action_upper_bound - self.action_lower_bound) / 16
        self.actions_variance = np.tile(
            np.expand_dims(np.expand_dims(self.actions_variance, 0), 0),
            [self.planning_horizon, self.num_agents, 1])

    def get_actions(self, current_states):
        best_cost = np.zeros([self.num_agents])
        best_cost = best_cost.fill(np.float('inf'))
        best_action = None
        iteration_index = 0
        current_actions_mean = np.array(self.actions_mean)
        current_actions_var = np.array(self.actions_variance)
        while iteration_index < self.max_iterations:
            print("cem iteration number ", iteration_index)
            #TODO: change to truncated one
            action_samples = np.random.normal(current_actions_mean,
                                              np.sqrt(current_actions_var),
                                              size=[self.population_size,
                                                    *self.actions_mean.shape])
            #shuffle some axis
            rewards = self.model.evaluate_trajectories(current_states,
                                                       action_samples) #num_of_agents, particles
            costs = -rewards
            elites_indicies = np.argpartition(costs, self.num_elite,
                                              axis=1)
            elites_indicies = elites_indicies[:, self.num_elite]
            print("current cost is ", np.min(costs, axis=1))
            # if np.min(costs, axis=1) < best_cost:
            elites = action_samples[elites_indicies]
            new_mean = np.mean(elites, axis=0)
            new_variance = np.var(elites, axis=0)
            current_actions_mean = (self.alpha * current_actions_mean) + (
                                   (1 - self.alpha) * new_mean)
            current_actions_var = (self.alpha * current_actions_var) + (
                    (1 - self.alpha) * new_variance)
            iteration_index += 1
        return current_actions_mean
