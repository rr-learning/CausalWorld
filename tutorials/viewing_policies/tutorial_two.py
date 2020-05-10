import causal_rl_bench.viewers.task_viewer as viewer
from causal_rl_bench.loggers.data_loader import DataLoader


def example():
    # This tutorial shows how to view logged episodes

    data = DataLoader(episode_directory='pushing_episodes')
    episode = data.get_episode(6)

    # Record a video of the logged episode is done in one line
    viewer.record_video_of_episode(episode=episode, file_name="pushing_video")

    # Similarly for interactive visualization in the GUI
    viewer.view_episode(episode)


if __name__ == '__main__':
    example()
