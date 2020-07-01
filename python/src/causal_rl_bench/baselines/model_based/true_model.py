from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np


class TrueModel(object):
    def __init__(self, _make_env_func, parallel_agents):
        self.parallel_agents = parallel_agents
        self.envs = SubprocVecEnv([_make_env_func() for i in
                                  range(self.parallel_agents)])
        return

    def evaluate_trajectories(self, action_sequences):
        #initial_state [0state_dim]
        #action_sequences number_of_[num_of_particles, horizon_length, action_dim]
        horizon_length = action_sequences.shape[1]
        num_of_particles = action_sequences.shape[0]
        rewards = np.zeros([num_of_particles])
        assert ((float(num_of_particles) /
                 self.parallel_agents).is_integer())
        for j in range(0, num_of_particles, self.parallel_agents):
            self.envs.reset()
            total_reward = np.zeros([self.parallel_agents])
            for k in range(horizon_length):
                actions = action_sequences[j:j+self.parallel_agents, k]
                task_observations, current_reward, done, info = \
                    self.envs.step(actions)
                total_reward += current_reward
            rewards[j:j+self.parallel_agents] = total_reward
        return rewards

    def end_sim(self):
        self.envs.close()
