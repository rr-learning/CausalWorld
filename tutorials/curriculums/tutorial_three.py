"""
This tutorial shows you how to create a curriculum using an intervention actor
which intervenes on the goal at specific time (start of the episode)
"""

from causal_world.task_generators.task import task_generator
from causal_world.envs.causalworld import CausalWorld
from causal_world.intervention_actors import RandomInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    #initialize env
    task = task_generator(task_generator_id='pick_and_place')
    env = CausalWorld(task, skip_frame=10, enable_visualization=True)

    # define a custom curriculum of interventions:
    # Goal intervention actor each episode after reset

    env = CurriculumWrapper(env,
                            intervention_actors=[RandomInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])

    for reset_idx in range(30):
        obs = env.reset()
        for time in range(300):
            obs, reward, done, info = env.step(env.action_space.low)
    env.close()


if __name__ == '__main__':
    example()
