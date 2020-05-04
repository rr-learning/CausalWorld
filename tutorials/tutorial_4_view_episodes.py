from causal_rl_bench.envs.world import World
from causal_rl_bench.loggers.world_logger import WorldLogger
from causal_rl_bench.viewers.task_viewer import TaskViewer
from causal_rl_bench.tasks.cuboid_silhouettes import CuboidSilhouette
import numpy as np
import time


def main():
    world_log = WorldLogger("output/logs/cuboid_silhouette.pickle")
    task_viewer = TaskViewer()

    # Record a specific episode
    task_viewer.record_animation_of_episode(world_log.episodes[2])

    # Record multiple episodes in individual files
    task_viewer.record_animation_of_episode(world_log.episodes[2:4])

    # Viewing a specific episode
    task_viewer.view_episode(world_log.episodes[2])

    # View multiple episodes one after another
    task_viewer.view_episode(world_log.episodes[2:4])


if __name__ == '__main__':
    main()
