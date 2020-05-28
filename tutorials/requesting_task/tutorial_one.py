from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import numpy as np
import time


def example():
    task = task_generator(task_generator_id='pushing')
    env = World(task=task, enable_visualization=True)
    for _ in range(200):
        obs = env.reset()
        # print(obs)
        chosen_intervention = env.do_single_random_intervention()
        # env.do_intervention({'tool_block': {'position':
        #                                         np.array([0, 0, 0])}})
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
            # print(obs)
            # print(reward)
            # print(done)
            # print(info)
    env.close()


if __name__ == '__main__':
    example()
