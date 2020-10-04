"""
This tutorial shows you how to control the allowed space for interventions
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import numpy as np


def without_intervention_split():
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention(
            {'stage_color': np.random.uniform(0, 1, [
                3,
            ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_1():
    task = generate_task(task_generator_id='pushing',
                          variables_space='space_a')
    env = CausalWorld(task=task, enable_visualization=False)
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention(
            {'stage_color': np.random.uniform(0, 1, [
                3,
            ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_2():
    task = generate_task(task_generator_id='pushing',
                          variables_space='space_b')
    env = CausalWorld(task=task, enable_visualization=False)
    interventions_space = task.get_intervention_space_a()
    env.reset()
    for _ in range(2):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal, obs = env.do_intervention({
            'stage_color':
                np.random.uniform(interventions_space['stage_color'][0],
                                  interventions_space['stage_color'][1])
        })
        print("Intervention success signal", success_signal)
    env.close()


if __name__ == '__main__':
    without_intervention_split()
    with_intervention_split_1()
    with_intervention_split_2()
