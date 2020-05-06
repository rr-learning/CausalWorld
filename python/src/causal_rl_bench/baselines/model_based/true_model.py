from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
import time


class TrueModel(object):
    def __init__(self, _make_env, num_of_envs,
                 num_of_particles, parallel_agents):
        self.num_of_envs = num_of_envs
        self.num_of_particles = num_of_particles
        self.parallel_agents = parallel_agents
        assert ((float(self.num_of_particles) /
                 self.parallel_agents).is_integer())
        #now we need to evaluate num_of_envs*num_of_particles
        self.envs = SubprocVecEnv([_make_env(rank=i) for i in
                                  range(self.parallel_agents)])
        self.envs.reset()
        return

    def evaluate_trajectories(self, initial_states, action_sequences):
        #initial_state 5, state_dim
        #action_sequences number_of_particles, horizon_length, 5, action_dim
        #repeat initial states for particles
        horizon_length = action_sequences.shape[1]
        rewards = np.zeros([self.num_of_envs, self.num_of_particles])
        for i in range(self.num_of_envs):
            for j in range(0, self.num_of_particles, self.parallel_agents):
                print("tryout now ")
                self.envs.env_method("set_full_state",
                                     initial_states[i])
                print("already set state ")
                for k in range(horizon_length):
                    actions = action_sequences[j:j+self.parallel_agents, k, i]
                    task_observations, current_reward, done, info = \
                        self.envs.step(actions)
                    rewards[i, j:j+self.parallel_agents] += current_reward
        return rewards
