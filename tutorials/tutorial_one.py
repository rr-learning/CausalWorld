from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
import numpy as np
import time
import gym
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
import matplotlib.pyplot as plt


def example():
    task = Task(task_id='pushing')
    env = World(task=task, skip_frame=35, enable_visualization=True,
                observation_mode="structured", enable_goal_image=False,
                normalize_observations=True)
    recorder = VideoRecorder(env,
                             'video.mp4')
    # current_state = env.get_full_state()
    for i in range(5):
        env.reset()
        # env.set_full_state(current_state)
        # env.do_random_intervention()
        for j in range(2000):
            recorder.capture_frame()
            # recorder.capture_frame()
            # env.step(
            #     np.random.uniform(env.action_space.low, env.action_space.high,
            #                       env.action_space.shape))
            start = time.time()
            obs, reward, done, info = env.step(
                np.random.uniform(env.action_space.low, env.action_space.high,
                                  env.action_space.shape))
            print(reward)
            end = time.time()
            # print(end - start)
            # env.render()

    recorder.capture_frame()
    recorder.close()
    env.close()


if __name__ == '__main__':
    example()
