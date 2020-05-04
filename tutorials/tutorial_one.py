from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.cuboid_silhouettes import CuboidSilhouette
import numpy as np
import time


def example():
    task = PushingTask()
    env = World(task=task, skip_frame=0.02, enable_visualization=True)
    env.reset()
    # current_state = env.get_full_state()
    for i in range(5):
        # env.reset()
        # env.set_full_state(current_state)
        env.do_random_intervention()
        for i in range(100):
            # env.step(
            #     np.random.uniform(env.action_space.low, env.action_space.high,
            #                       env.action_space.shape))
            env.step(np.zeros(shape=[9,]))

    # Switching to a counterfactual variant of the task
    # env.switch_task(task.get_counterfactual_variant(cube_color="green", unit_length=0.08))
    # env.reset()
    # for i in range(5):
    #     env.reset()
    #     for i in range(100):
    #         env.step(
    #             np.random.uniform(env.action_space.low, env.action_space.high,
    #                               env.action_space.shape))
    env.close()


if __name__ == '__main__':
    example()
