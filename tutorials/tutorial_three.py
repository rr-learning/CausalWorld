from causal_rl_bench.envs.world import World
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.tasks.task import Task
import numpy as np
import time

import matplotlib.pyplot as plt


def example():
    task = Task(task_id='cuboid_silhouette', silhouette_size=[3, 2, 1])
    data_recorder = DataRecorder(rec_dumb_frequency=10)  # default rec_dumb_frequency is 1000
    env = World(task=task,
                skip_frame=20,
                seed=1,
                observation_mode="cameras",
                normalize_observations=True,
                enable_visualization=True)
    for i in range(20):
        obs = env.reset()
        for i in range(50):
            obs, _, _, _ = env.step(np.random.uniform(env.action_space.sample()))
    env.close()


if __name__ == '__main__':
    example()