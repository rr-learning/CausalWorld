"""
This tutorial shows you how to intervene during an episode to for example
create a guided goal environment.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def example():
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    counter = 0
    for _ in range(1):
        for i in range(210):
            obs, reward, done, info = env.step(env.action_space.low)
            if i % 50 == 0 and i > 0:
                print(i)
                intervention = {'goal_block': {'cartesian_position':
                                                   [0, -0.08+(0.04*counter),
                                                    0.0325],
                                               'color':[0, 0, 1]}}
                env.do_intervention(intervention, check_bounds=False)
                counter += 1
                print("intervention")
            if i == 201:
                intervention = {'goal_block': {
                    'cartesian_position': [0, 0.08,  0.0325],
                    'color': [0, 1, 0]}}
                env.do_intervention(intervention, check_bounds=False)
    env.close()


if __name__ == '__main__':
    example()

