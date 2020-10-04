from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np


class SimulatorModel(object):

    def __init__(self, _make_env_func, parallel_agents):
        """
        This class instantiates a dynamics model based on the pybullet simulator
        (i.e: simulates exactly the result of the actions), it can be used
        for reward tuning and verifying tasks..etc

        :param _make_env_func: (func) a function if called it will return a gym
                                      environment.
        :param parallel_agents: (int) number of parallel agents to siumulate
                                      to evaluate the actions.
        """
        self.parallel_agents = parallel_agents
        self.envs = SubprocVecEnv(
            [_make_env_func() for i in range(self.parallel_agents)])
        return

    def evaluate_trajectories(self, action_sequences):
        """
        A function to be called to evaluate the action sequences and return
        the corresponding reward for each sequence.

        :param action_sequences: (nd.array) actions to be evaluated
                                            (number of sequences, horizon length)
        :return: (nd.array) sum of rewards for each action sequence.
        """
        horizon_length = action_sequences.shape[1]
        num_of_particles = action_sequences.shape[0]
        rewards = np.zeros([num_of_particles])
        assert ((float(num_of_particles) / self.parallel_agents).is_integer())
        for j in range(0, num_of_particles, self.parallel_agents):
            self.envs.reset()
            total_reward = np.zeros([self.parallel_agents])
            for k in range(horizon_length):
                actions = action_sequences[j:j + self.parallel_agents, k]
                task_observations, current_reward, done, info = \
                    self.envs.step(actions)
                total_reward += current_reward
            rewards[j:j + self.parallel_agents] = total_reward
        return rewards

    def end_sim(self):
        """
        Closes the environments that were used for simulation.
        :return:
        """
        self.envs.close()
        return
