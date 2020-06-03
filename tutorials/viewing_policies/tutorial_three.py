from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
from causal_rl_bench.loggers.data_loader import DataLoader


def example():
    # This tutorial shows how to view a random policy on the pyramid task

    task = task_generator(task_generator_id='pyramid')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 200

    viewer.record_video_of_random_policy(task=task,
                                         world_params=world_params,
                                         file_name="pyramid_video",
                                         number_of_resets=1,
                                         max_time_steps=300)


if __name__ == '__main__':
    example()