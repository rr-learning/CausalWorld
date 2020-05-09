from stable_baselines.common import set_global_seeds
from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.baselines.model_based.true_model import TrueModel
from causal_rl_bench.baselines.model_based.optimizers.cem import \
    CrossEntropyMethod
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from causal_rl_bench.tasks.task import Task

seed = 0
skip_frame = 35
num_of_particles = 250
num_elite = 25
max_iterations = 5
horizon_length = 6
parallel_agents = 25


def _make_env():
    def _init():
        task = PushingTask()
        env = World(task=task, skip_frame=skip_frame,
                    enable_visualization=False,
                    seed=seed)
        return env
    set_global_seeds(seed)
    return _init


def run_mpc():
    task = Task(task_id='pushing')
    env = World(task=task, skip_frame=skip_frame, enable_visualization=False,
                seed=seed)
    recorder = VideoRecorder(env, 'pushing.mp4')
    env.reset()
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
    current_state = env.get_full_state()
    actions = optimizer.get_actions(current_state)
    true_model.end_sim()
    env.set_full_state(current_state)
    for i in range(horizon_length):
        recorder.capture_frame()
        env.step(actions[i])
    recorder.capture_frame()
    recorder.close()
    env.close()


if __name__ == '__main__':
    run_mpc()

