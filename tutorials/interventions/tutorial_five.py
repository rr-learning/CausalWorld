from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import numpy as np


def example():
    task = task_generator(task_generator_id='picking')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset({'goal_block': {'cartesian_position': [0.1, 0.1, 0.1]}})
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.reset_default_state()
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.reset()
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
