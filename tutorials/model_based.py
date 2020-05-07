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


skip_frame = 35


def _make_env(rank):
    def _init():
        task = PushingTask()
        env = World(task=task, skip_frame=skip_frame,
                    enable_visualization=False,
                    seed=seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def train_policy(num_of_envs):
    #plan for the next horizon
    task = PushingTask()
    env = World(task=task, skip_frame=skip_frame, enable_visualization=False,
                seed=0)
    recorder = VideoRecorder(env,
                             'pick_up.mp4')
    env.reset()
    num_of_particles = 500
    horizon_length = 6
    parallel_agents = 1
    num_elite = 50
    max_iterations = 32
    true_model = TrueModel(_make_env,
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
                                   model=true_model)
    current_state = env.get_full_state()
    actions = optimizer.get_actions(current_state)
    env.set_full_state(current_state)
    for i in range(horizon_length):
        #TODO: set quick state?
        recorder.capture_frame()
        env.step(actions[i])
    recorder.capture_frame()
    recorder.close()
    env.close()
    true_model.end_sim()
    return


if __name__ == '__main__':
    train_policy(1)

