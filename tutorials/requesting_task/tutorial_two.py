"""
This tutorial shows you how to generate a task using one of the task generators
and then using it with the CausalWorld environment but having camera observations
instead of the default structured observations.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
import matplotlib.pyplot as plt


def example():
    task = generate_task(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task,
                      skip_frame=10,
                      enable_visualization=True,
                      seed=0,
                      action_mode="joint_positions",
                      observation_mode="pixel",
                      camera_indicies=[0, 1, 2])
    env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
    #show last images
    for i in range(6):
        plt.imshow(obs[i])
        plt.show()
    env.close()


if __name__ == '__main__':
    example()
