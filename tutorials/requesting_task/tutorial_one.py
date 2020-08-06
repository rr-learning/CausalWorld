"""
This tutorial shows you how to generate a task using one of the task generators
and then using it with the CausalWorld environment.
"""
from causal_world import CausalWorld
from causal_world import task_generator


def example():
    task = task_generator(task_generator_id='reaching')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(20):
        env.reset()
        for _ in range(1000):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
