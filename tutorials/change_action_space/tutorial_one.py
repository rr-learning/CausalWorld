from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import task_generator
from causal_world.wrappers.action_wrappers import DeltaActionEnvWrapper
import numpy as np


def apply_delta_action():
    task = task_generator(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      action_mode="joint_positions",
                      normalize_actions=True,
                      normalize_observations=True,
                      skip_frame=1)
    env = DeltaActionEnvWrapper(env)
    for _ in range(50):
        obs = env.reset()
        for _ in range(1000):
            desired_action = np.zeros([
                9,
            ])
            obs, reward, done, info = env.step(desired_action)
    env.close()


if __name__ == '__main__':
    apply_delta_action()
