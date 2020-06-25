from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import numpy as np


def example():
    task = task_generator(task_generator_id='pushing')
    env = World(task=task, enable_visualization=True)
    for _ in range(10):
        env.reset()
        for _ in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
