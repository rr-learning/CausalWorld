from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np


class TrueModel(object):
    def __init__(self, _make_env, num_of_envs, num_of_particles):
        self.num_of_envs = num_of_envs
        self.num_of_particles = num_of_particles
        self.envs = SubprocVecEnv([_make_env(rank=i) for i in
                                  range(self.num_of_envs*self.num_of_particles)])
        self.envs.reset()
        return

    def evaluate_trajectories(self, initial_states, action_sequences):
        #initial_state 5, state_dim
        #action_sequences number_of_particles, horizon_length, 5, action_dim
        #repeat initial states for particles
        horizon_length = action_sequences.shape[1]
        for i in range(self.num_of_envs):
            env_indices = list(np.arange(i*self.num_of_particles,
                               (i+1)*self.num_of_particles))
            self.envs.env_method("set_full_state",
                                 initial_states[i], indices=env_indices)
        rewards = np.zeros([self.num_of_envs, self.num_of_particles])
        for i in range(horizon_length):
            actions_reshaped = np.reshape(action_sequences[:, i],
                                          [self.num_of_envs *
                                           self.num_of_particles, -1])
            task_observations, current_reward, done, info = \
                self.envs.step(actions_reshaped)
            rewards_reshaped = np.reshape(current_reward,
                                          [self.num_of_envs,
                                           self.num_of_particles])
            rewards += rewards_reshaped
        return rewards
