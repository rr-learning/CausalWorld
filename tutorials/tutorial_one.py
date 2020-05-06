from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.cuboid_silhouettes import CuboidSilhouette
import numpy as np
import time
import gym
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time


def example():
    task = PushingTask()
    #TODO: modify skip frame
    env = World(task=task, skip_frame=20, enable_visualization=True)
    recorder = VideoRecorder(env,
                             'video.mp4')
    env.reset()
    # current_state = env.get_full_state()
    for i in range(2):
        # env.reset()
        # env.set_full_state(current_state)
        env.do_random_intervention()
        for j in range(100):
            recorder.capture_frame()
            # recorder.capture_frame()
            # env.step(
            #     np.random.uniform(env.action_space.low, env.action_space.high,
            #                       env.action_space.shape))
            start = time.time()
            env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))
            end = time.time()
            print(end - start)
            # env.render()


    # Switching to a counterfactual variant of the task
    # env.switch_task(task.get_counterfactual_variant(cube_color="green", unit_length=0.08))
    # env.reset()
    # for i in range(5):
    #     env.reset()
    #     for i in range(100):
    #         env.step(
    #             np.random.uniform(env.action_space.low, env.action_space.high,
    #                               env.action_space.shape))
    recorder.capture_frame()
    recorder.close()
    env.close()


if __name__ == '__main__':
    example()
