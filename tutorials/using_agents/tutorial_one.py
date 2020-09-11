"""
This tutorial shows you how to use a random policy with any env.
"""
from causal_world import CausalWorld
from causal_world import task_generator
from causal_world.actors import GraspingPolicy
from causal_world.viewers import record_video
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers import CurriculumWrapper


def example():
    task = task_generator(task_generator_id='stacking2')
    env = CausalWorld(task=task, enable_visualization=True,
                      action_mode='end_effector_positions')
    env = CurriculumWrapper(env, intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(1, 100, 1, 0)])
    policy = GraspingPolicy([0, 1])
    record_video(env=env,
                 policy=policy,
                 file_name="stacking2",
                 number_of_resets=2,
                 max_time_steps=1500)
    env.close()


if __name__ == '__main__':
    example()
