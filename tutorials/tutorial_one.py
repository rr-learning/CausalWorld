from causal_rl_bench.envs.world import World
import numpy as np
import time


def example():
    env = World(control_rate=0.001, enable_visualization=True)
    env.reset()
    for i in range(5):
        env.reset()
        for i in range(2000):
            env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))
    time.sleep(20)


if __name__ == '__main__':
    example()