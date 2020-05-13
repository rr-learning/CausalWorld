from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
import numpy as np


def example():
    task = Task(task_id='pushing')
    env = World(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2000):
        env.reset()
        obs, reward, done, info = env.step(np.zeros([9,]))
    env.close()


if __name__ == '__main__':
    example()
