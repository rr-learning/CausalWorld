"""
This tutorial shows you how to perform a random intervention on one of the
variables in the environment.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def example():
    task = generate_task(task_generator_id='picking')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(50):
        random_intervention_dict, success_signal, obs = \
            env.do_single_random_intervention()
        print("The random intervention performed is ",
              random_intervention_dict)
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
