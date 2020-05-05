from causal_rl_bench.envs.world import World
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.tasks.cuboid_silhouettes import CuboidSilhouette
import numpy as np
import time


def example():
    task = CuboidSilhouette()
    data_recorder = DataRecorder(rec_dumb_frequency=10)  # default rec_dumb_frequency is 1000
    env = World(task=task,
                skip_frame=0.02,
                enable_visualization=True,
                data_recorder=data_recorder)
    for i in range(35):
        env.reset()
        # env.do_random_intervention()
        for i in range(50):
            env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))
    env.close()


if __name__ == '__main__':
    example()