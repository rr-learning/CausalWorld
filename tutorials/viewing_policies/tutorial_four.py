"""
This tutorial shows you how to to view policies of trained actors.
"""

from causal_world.task_generators.task import generate_task
from stable_baselines import SAC
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import causal_world.viewers.task_viewer as viewer


def example():
    # This tutorial shows how to view policies of trained actors

    task = generate_task(task_generator_id='picking')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 0
    stable_baselines_policy_path = "./model_2000000_steps.zip"
    model = SAC.load(stable_baselines_policy_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    # # Record a video of the policy is done in one line
    viewer.record_video_of_policy(task=task,
                                  world_params=world_params,
                                  policy_fn=policy_fn,
                                  file_name="pushing_video",
                                  number_of_resets=10,
                                  max_time_steps=10 * 100)

    # Similarly for interactive visualization in the GUI
    viewer.view_policy(task=task,
                       world_params=world_params,
                       policy_fn=policy_fn,
                       max_time_steps=40 * 600,
                       number_of_resets=40)


if __name__ == '__main__':
    example()
