"""
This tutorial shows you how to to view policies of trained actors.
"""

from causal_world.task_generators.task import generate_task
from stable_baselines import PPO2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import causal_world.viewers.task_viewer as viewer
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    # This tutorial shows how to view policies of trained actors

    task = generate_task(task_generator_id='pick_and_place')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 0
    stable_baselines_policy_path = "./model_100000000_steps.zip"
    model = PPO2.load(stable_baselines_policy_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    # Similarly for interactive visualization in the GUI
    viewer.view_policy(task=task,
                       world_params=world_params,
                       policy_fn=policy_fn,
                       max_time_steps=40 * 600,
                       number_of_resets=40,
                       env_wrappers=[CurriculumWrapper],
                       env_wrappers_args=[{'intervention_actors':[GoalInterventionActorPolicy()],
                                           'actives': [(0, 1000000000, 1, 0)]}])


if __name__ == '__main__':
    example()
