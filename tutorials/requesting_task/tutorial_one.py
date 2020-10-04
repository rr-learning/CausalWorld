"""
This tutorial shows you how to generate a task using one of the task generators
and then using it with the CausalWorld environment.
"""
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task


def example():
    task = generate_task(task_generator_id='creative_stacked_blocks')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(20):
        env.reset()
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
