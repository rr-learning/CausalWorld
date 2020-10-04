"""
This tutorial shows you how to discretize the action space of the robot itself,
by using the object selector wrapper, first position chooses the object,
second position chooses up, down, right, left, up, stay and third position is
for rotation.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.wrappers.planning_wrappers import ObjectSelectorWrapper


def example():
    task = generate_task(task_generator_id='picking')
    env = CausalWorld(task=task, enable_visualization=True)
    env = ObjectSelectorWrapper(env)
    for _ in range(50):
        obs = env.reset()
        #go up
        for i in range(70):
            obs, reward, done, info = env.step([0, 1, 0])
        # rotate yaw
        for i in range(20):
            obs, reward, done, info = env.step([0, 0, 1])
        for i in range(50):
            obs, reward, done, info = env.step([0, 5, 0])
        for i in range(20):
            obs, reward, done, info = env.step([0, 0, 1])
            # print(obs)
        for i in range(50):
            obs, reward, done, info = env.step([0, 2, 0])
            # print(obs)
    env.close()


if __name__ == '__main__':
    example()
