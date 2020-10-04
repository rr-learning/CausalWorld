"""
This tutorial shows you how to create a curriculum using several intervention
actors which intervenes on the environment at specific times and in different
way, by intervening on different variables.
"""

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.intervention_actors import GoalInterventionActorPolicy, VisualInterventionActorPolicy, \
    RandomInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    #initialize env
    task_gen = generate_task(task_generator_id='pushing')
    env = CausalWorld(task_gen, skip_frame=10, enable_visualization=True)

    # define a custom curriculum of interventions:

    # No intervention actor is defined until episode number 5
    # Goal intervention actor from episode number 5 to 10 after reset at time step 0
    # Visual intervention actor from episode number 10 to 20 every two episodes after reset at time step 0
    # Random intervention actor from episode number 20 to 25 after reset at time step 0
    # Goal intervention actor from episode number 25 to 30 each at time step 50

    env = CurriculumWrapper(env,
                            intervention_actors=[
                                GoalInterventionActorPolicy(),
                                VisualInterventionActorPolicy(),
                                RandomInterventionActorPolicy(),
                                GoalInterventionActorPolicy()
                            ],
                            actives=[(5, 10, 1, 0), (10, 20, 2, 0),
                                     (20, 25, 1, 0), (25, 30, 1, 50)])

    for reset_idx in range(30):
        obs = env.reset()
        for time in range(100):
            desired_action = env.action_space.sample()
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    example()
