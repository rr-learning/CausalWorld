from causal_rl_bench.loggers.data_loader import DataLoader
from causal_rl_bench.viewers.task_viewer import TaskViewer


def main():
    data = DataLoader(data_path="output/logs/episode_0_9")
    task_viewer = TaskViewer()

    # Record a specific episode
    task_viewer.record_animation_of_episode(data.get_episodes()[4], num=4)

    # Record a specific episode from another file
    data.load_data(data_path="output/logs/episode_10_19")
    task_viewer.record_animation_of_episode(data.get_episodes()[4], num=14)

    # Viewing a specific episode
    task_viewer.view_episode(data.get_episodes()[3])


if __name__ == '__main__':
    main()
