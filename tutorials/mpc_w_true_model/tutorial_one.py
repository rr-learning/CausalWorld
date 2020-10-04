"""
This tutorial shows you how to use Model Predictive Control
with the true model.
"""
from stable_baselines.common import set_global_seeds
from causal_world.envs.causalworld import CausalWorld
from causal_world.dynamics_model import SimulatorModel
from causal_world.utils.mpc_optimizers import \
    CrossEntropyMethod
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from causal_world.task_generators.task import generate_task

seed = 0
skip_frame = 35
num_of_particles = 500
num_elite = 50
max_iterations = 20
horizon_length = 6
parallel_agents = 25


def _make_env():

    def _init():
        task = generate_task(
            task_generator_id='picking',
            joint_positions=[-0.21737874, 0.55613149,
                             -1.09308519, -0.12868997,
                             0.52551013, -1.08006493,
                             -0.00221536, 0.46163487,
                             -1.00948735],
            tool_block_position=[0.0, 0, 0.035],
            fractional_reward_weight=1,
            dense_reward_weights=np.array([0, 10, 0,
                                           1, 1, 0, 0,
                                           0]))
        env = CausalWorld(task=task,
                          skip_frame=skip_frame,
                          enable_visualization=False,
                          seed=seed)
        return env

    set_global_seeds(seed)
    return _init


def run_mpc():
    task = generate_task(
        task_generator_id='picking',
        joint_positions=[-0.21737874, 0.55613149,
                         -1.09308519, -0.12868997,
                         0.52551013, -1.08006493,
                         -0.00221536, 0.46163487,
                         -1.00948735],
        tool_block_position=[0.0, 0, 0.035],
        fractional_reward_weight=1,
        dense_reward_weights=np.array([0, 10, 0,
                                       1, 1, 0, 0,
                                       0]))
    env = CausalWorld(task=task,
                      skip_frame=1,
                      enable_visualization=False,
                      seed=seed)
    true_model = SimulatorModel(_make_env, parallel_agents=parallel_agents)
    optimizer = CrossEntropyMethod(
        planning_horizon=horizon_length,
        max_iterations=max_iterations,
        population_size=num_of_particles,
        num_elite=num_elite,
        action_upper_bound=np.array(env.action_space.high),
        action_lower_bound=np.array(env.action_space.low),
        model=true_model)
    env.reset()
    actions = optimizer.get_actions()
    true_model.end_sim()
    recorder = VideoRecorder(env, 'picking.mp4')
    for i in range(horizon_length):
        for _ in range(skip_frame):
            recorder.capture_frame()
            obs, reward, done, info = env.step(actions[i])
    recorder.capture_frame()
    recorder.close()
    env.close()


if __name__ == '__main__':
    run_mpc()
