from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import task_generator
import numpy as np


def example():
    task = task_generator(task_generator_id="creative_stacked_blocks")
    env = CausalWorld(task=task, enable_visualization=False, seed=0)
    actions = [env.action_space.sample() for _ in range(200)]
    env.reset()
    observations_1 = []
    rewards_1 = []
    for i in range(200):
        observations, rewards, _, _ = env.step(actions[i])
        if i == 100:
            state = env.get_state()
        observations_1.append(observations)
        rewards_1.append(rewards)
    env.set_state(state)
    for i in range(101, 200):
        observations, rewards, _, _ = env.step(actions[i])
        assert np.array_equal(observations_1[i], observations)
    env.close()


if __name__ == '__main__':
    example()
