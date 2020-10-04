"""
This tutorial shows you how to get the full state from the current
environment.
"""
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task


def example():
    task = generate_task(task_generator_id='creative_stacked_blocks')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(1):
        env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(env.action_space.sample())
    print(env.get_current_state_variables())
    env.close()


if __name__ == '__main__':
    example()
