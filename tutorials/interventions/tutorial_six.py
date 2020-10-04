"""
This tutorial shows you how to use the intervention space is sampling an intervention
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np


def example():
    task = generate_task(task_generator_id='pick_and_place')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    intervention_space = env.get_variable_space_used()
    for _ in range(100):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.low)
        intervention = {'tool_block': {'size': np.random.uniform(intervention_space['tool_block']['size'][0],
                                                                 intervention_space['tool_block']['size'][1])}}
        env.do_intervention(intervention)
    env.close()


if __name__ == '__main__':
    example()

