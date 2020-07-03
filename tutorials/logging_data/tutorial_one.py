from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator

from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.loggers.data_loader import DataLoader

import causal_rl_bench.viewers.task_viewer as viewer


def example():
    # Here you learn how to record/ log entire episodes into a directory
    # to reuse it later e.g. for reviewing logged episodes or using this
    # data for pre-training  policies.

    # Construct a data_recorder that keeps track of every change in the environment
    # We set the recording dumb frequency of episodes into log_files to 11 (default is 100)
    data_recorder = DataRecorder(output_directory='pushing_episodes', rec_dumb_frequency=11)

    # Pass the data recorder to the World
    task = task_generator(task_generator_id='pushing')
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
    task = task_generator(episode.task_name, **episode.task_params)
    env = CausalWorld(task,
                      **episode.world_params,
                      enable_visualization=True)
    env.set_starting_state(episode.initial_full_state)
    env.reset()
    for action in episode.robot_actions:
        env.step(action)
    env.close()

    # You can achieve the same by using the viewer module in one line
    viewer.view_episode(episode)

    # TODO: I get an issue with closing the environment and it gets rendered from a strange camera view. Doo you get the same error?


if __name__ == '__main__':
    example()