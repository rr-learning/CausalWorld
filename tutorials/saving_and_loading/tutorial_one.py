from causal_rl_bench.utils.config_utils import load_world
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from causal_rl_bench.intervention_agents import VisualInterventionActorPolicy
from causal_rl_bench.curriculum import InterventionsCurriculum
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.wrappers import DeltaAction
import numpy as np


def example():
    #initialize env
    task_gen = task_generator(task_generator_id='pushing')
    env = World(task_gen, skip_frame=1,
                enable_visualization=True)
    env = DeltaAction(env)
    env = CurriculumWrapper(env,
                            intervention_actors=[
                                VisualInterventionActorPolicy()],
                            episodes_hold=[3],
                            timesteps_hold=[None])

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