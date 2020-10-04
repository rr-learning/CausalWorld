"""
This tutorial shows you how to save the world and load it again with all
its wrappers as well.
"""
from causal_world.utils.config_utils import load_world
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.intervention_actors import VisualInterventionActorPolicy
from causal_world.curriculum import Curriculum
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.wrappers import DeltaActionEnvWrapper
import numpy as np


def example():
    #initialize env
    task_gen = generate_task(task_generator_id='pushing')
    env = CausalWorld(task_gen, skip_frame=1, enable_visualization=True)
    env = DeltaActionEnvWrapper(env)
    env = CurriculumWrapper(
        env,
        intervention_actors=[VisualInterventionActorPolicy()],
        actives=[(0, 20, 1, 0)])

    for reset_idx in range(10):
        obs = env.reset()
        for time in range(15):
            obs, reward, done, info = env.step(action=np.zeros(9,))
    env.save_world('./')
    env.close()
    #now load it again

    env = load_world(tracker_relative_path='./', enable_visualization=True)
    for reset_idx in range(10):
        obs = env.reset()
        for time in range(15):
            obs, reward, done, info = env.step(action=np.zeros(9,))


if __name__ == '__main__':
    example()
