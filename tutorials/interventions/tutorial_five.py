"""
This tutorial shows you how to intervene on the default starting state,
"""
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np


def example():
    task = generate_task(task_generator_id='picking')
    env = CausalWorld(task=task, enable_visualization=True)
    env.set_starting_state(
        {'goal_block': {
            'cartesian_position': [0.1, 0.1, 0.1]
        }})
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.reset_default_state()
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.reset()
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
