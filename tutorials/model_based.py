import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.baselines.model_based.true_model import TrueModel
from causal_rl_bench.baselines.model_based.optimizers.cem import \
    CrossEntropyMethod
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

seed = 0


skip_frame = 50


def _make_env(rank):
    def _init():
        task = PushingTask()
        env = World(task=task, skip_frame=skip_frame,
                    enable_visualization=False,
                    seed=seed + rank)
        env.enforce_max_episode_length(episode_length=50)
        return env
    set_global_seeds(seed)
    return _init


def train_policy(num_of_envs):
    #plan for the next horizon
    task = PushingTask()
<<<<<<< Updated upstream
    env = World(task=task, skip_frame=skip_frame, enable_visualization=False)
    recorder = VideoRecorder(env,
                             'hi.mp4')
=======
    env = World(task=task, skip_frame=20, enable_visualization=False)
>>>>>>> Stashed changes
    env.reset()
    num_of_particles = 1000
    horizon_length = 100
    parallel_agents = 50
    num_elite = 100
    max_iterations = 5
    true_model = TrueModel(_make_env, num_of_envs,
                           num_of_particles=num_of_particles,
                           parallel_agents=parallel_agents)
    optimizer = CrossEntropyMethod(planning_horizon=horizon_length,
                                   max_iterations=max_iterations,
                                   population_size=num_of_particles,
                                   num_elite=num_elite,
                                   action_upper_bound=
                                   np.array(env.action_space.high),
                                   action_lower_bound=
                                   np.array(env.action_space.low),
                                   model=true_model,
                                   num_agents=num_of_envs)
    current_states = np.expand_dims(env.get_full_state(), 0)
    actions = optimizer.get_actions(current_states)
    for i in range(horizon_length):
        #TODO: set quick state?
        recorder.capture_frame()
        env.step(actions[i, 0, :])
    recorder.capture_frame()
    recorder.close()
    env.close()
    return


if __name__ == '__main__':
    train_policy(1)

