from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import Task
import numpy as np
import time


def example():
    task = Task(task_id='reaching')
    env = World(task=task, enable_visualization=True)
    for _ in range(200):
        obs = env.reset()
        print(obs)
        time.sleep(1)
        env.do_single_random_intervention()
        env.do_intervention({'floor_color': np.array([1, 0.5, 1])})
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
            print(obs)
            print(reward)
            print(done)
            print(info)
    env.close()


if __name__ == '__main__':
    example()
