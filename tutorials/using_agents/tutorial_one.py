"""
This tutorial shows you how to use a trained actor to solve pick and place.
"""
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from causal_world.actors import PickAndPlaceActorPolicy
from causal_world.viewers import record_video


def example():
    task = generate_task(task_generator_id='pick_and_place')
    env = CausalWorld(task=task, skip_frame=3,
                      enable_visualization=True)
    policy = PickAndPlaceActorPolicy()
    # record_video(env,
    #              policy,
    #              'pick_and_place',
    #               number_of_resets=1,
    #               max_time_steps=None)
    env.close()


if __name__ == '__main__':
    example()
