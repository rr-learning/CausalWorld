from stable_baselines.common import set_global_seeds
from causal_rl_bench.envs.world import World
from causal_rl_bench.baselines.model_based.true_model import TrueModel
from causal_rl_bench.baselines.model_based.optimizers.cem import \
    CrossEntropyMethod
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from causal_rl_bench.task_generators.task import task_generator

seed = 0
skip_frame = 35
num_of_particles = 250
num_elite = 12
max_iterations = 10
horizon_length = 8
parallel_agents = 25
randomize_joint_positions = False
randomize_block_pose = False
randomize_goal_block_pose = True


def _make_env():
    def _init():
        task = task_generator(task_generator_id='pushing',
                              randomize_joint_positions=randomize_joint_positions,
                              randomize_block_pose=randomize_block_pose,
                              randomize_goal_block_pose=randomize_goal_block_pose)
        env = World(task=task, skip_frame=skip_frame,
                    enable_visualization=False,
                    seed=seed)
        return env
    set_global_seeds(seed)
    return _init


def run_mpc():
    task = task_generator(task_generator_id='pushing', randomize_joint_positions=randomize_joint_positions,
                          randomize_block_pose=randomize_block_pose,
                          randomize_goal_block_pose=randomize_goal_block_pose)
    env = World(task=task, skip_frame=1, enable_visualization=False,
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
    current_state = env.get_full_state()
    actions = optimizer.get_actions(current_state)
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

