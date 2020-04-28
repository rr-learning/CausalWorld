from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.pushing import PushingTask
import numpy as np
import time


def example():
    task = PushingTask()
    env = World(task=task, control_rate=0.02, enable_visualization=True)
    env.reset()
    for i in range(5):
        env.reset()
        for i in range(100):
            env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))



if __name__ == '__main__':
    example()