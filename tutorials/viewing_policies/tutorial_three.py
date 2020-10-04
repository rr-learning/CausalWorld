"""
This tutorial shows you how to record a video of a random policy in the world.
"""

from causal_world.task_generators.task import generate_task
import causal_world.viewers.task_viewer as viewer
from causal_world.loggers.data_loader import DataLoader


def example():
    # This tutorial shows how to view a random policy on the pyramid task

    task = generate_task(task_generator_id='picking')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 200

    viewer.record_video_of_random_policy(task=task,
                                         world_params=world_params,
                                         file_name="picking_video",
                                         number_of_resets=1,
                                         max_time_steps=300)


if __name__ == '__main__':
    example()
