from stable_baselines.common import set_global_seeds
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.baselines.model_based.true_model import TrueModel
from causal_rl_bench.baselines.model_based.optimizers.cem import \
    CrossEntropyMethod
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from causal_rl_bench.task_generators.task import task_generator


seed = 0
skip_frame = 3
num_of_particles = 500
num_elite = 50
max_iterations = 5
horizon_length = 16
parallel_agents = 25


def _make_env():
    def _init():
        task = task_generator(task_generator_id='reaching',
                              joint_positions=[0., -0.5, -0.6,
                                               0., -0.4, -0.7,
                                               0., -0.4, -0.7])
        env = CausalWorld(task=task, skip_frame=skip_frame,
                          enable_visualization=False,
                          seed=seed)
        return env
    set_global_seeds(seed)
    return _init


def run_mpc():
    task = task_generator(task_generator_id='reaching',
                          joint_positions=[0., -0.5, -0.6,
                                           0., -0.4, -0.7,
                                           0., -0.4, -0.7])
    env = CausalWorld(task=task, skip_frame=skip_frame,
                      enable_visualization=False,
                      seed=seed)
    true_model = TrueModel(_make_env, parallel_agents=parallel_agents)
    optimizer = CrossEntropyMethod(planning_horizon=horizon_length,
                                   max_iterations=max_iterations,
                                   population_size=num_of_particles,
                                   num_elite=num_elite,
                                   action_upper_bound=
                                   np.array(env.action_space.high),
                                   action_lower_bound=
                                   np.array(env.action_space.low),
                                   model=true_model)
    env.reset()
    actions = optimizer.get_actions()
    true_model.end_sim()
    recorder = VideoRecorder(env, 'pushing.mp4')
    for i in range(horizon_length):
        for j in range(skip_frame):
            recorder.capture_frame()
            obs, reward, done, info = env.step(actions[i])
    recorder.capture_frame()
    recorder.close()
    env.close()


if __name__ == '__main__':
    run_mpc()
