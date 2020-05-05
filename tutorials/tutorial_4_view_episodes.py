from causal_rl_bench.envs.world import World
from causal_rl_bench.loggers.world_logger import WorldLogger
from causal_rl_bench.viewers.task_viewer import TaskViewer
from causal_rl_bench.tasks.cuboid_silhouettes import CuboidSilhouette
import numpy as np
import time


def main():
    world_log = WorldLogger(filename="cuboid_silhouette")
    task_viewer = TaskViewer()

    # Record multiple episodes in individual files
    task_viewer.record_animation_batch_of_episodes(world_log.episodes[:2])

    # Record a specific episode
    task_viewer.record_animation_of_episode(world_log.episodes[1], num=3)

    # View multiple episodes one after another
    task_viewer.view_batch_of_episodes(world_log.episodes[0:3])

    # Viewing a specific episode
    task_viewer.view_episode(world_log.episodes[4])


if __name__ == '__main__':
    main()
