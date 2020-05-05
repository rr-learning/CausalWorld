from causal_rl_bench.envs.world import World
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.tasks.cuboid_silhouette import CuboidSilhouette
import numpy as np
import time


def example():
    task = CuboidSilhouette(silhouette_size=[1, 2, 1])
    data_recorder = DataRecorder(rec_dumb_frequency=10)  # default rec_dumb_frequency is 1000
    env = World(task=task,
                skip_frame=20,
                enable_visualization=True)
    for i in range(10):
        env.reset()
        for i in range(200):
            env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))
    env.close()


if __name__ == '__main__':
    example()