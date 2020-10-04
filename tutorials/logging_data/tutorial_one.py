"""
This tutorial shows you how to use a data recorder to record some data for
imitation learning for instance and how to load the data again. Or replay some
episodes.
"""
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task

from causal_world.loggers.data_recorder import DataRecorder
from causal_world.loggers.data_loader import DataLoader

import causal_world.viewers.task_viewer as viewer


def example():
    # Here you learn how to record/ log entire episodes into a directory
    # to reuse it later e.g. for reviewing logged episodes or using this
    # data for pre-training  policies.

    # Construct a data_recorder that keeps track of every change in the environment
    # We set the recording dumb frequency of episodes into log_files to 11 (default is 100)
    data_recorder = DataRecorder(output_directory='pushing_episodes',
                                 rec_dumb_frequency=11)

    # Pass the data recorder to the World
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      data_recorder=data_recorder)

    # Record some episodes
    for _ in range(23):
        env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())
    env.close()

    # Load the logged episodes
    data = DataLoader(episode_directory='pushing_episodes')
    episode = data.get_episode(14)

    # Initialize a new environment according a specific episode and replay it
    task = generate_task(episode.task_name, **episode.task_params)
    env = CausalWorld(task, **episode.world_params, enable_visualization=True)
    env.set_starting_state(episode.initial_full_state,
                           check_bounds=False)
    for action in episode.robot_actions:
        env.step(action)
    env.close()

    # You can achieve the same by using the viewer module in one line
    viewer.view_episode(episode)


if __name__ == '__main__':
    example()
