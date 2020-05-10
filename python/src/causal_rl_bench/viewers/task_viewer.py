from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_rl_bench.tasks.task import Task
import os


def get_world(task_id, task_params, world_params, enable_visualization=False):
    if task_params is None:
        task = Task(task_id)
    else:
        task = Task(task_id, **task_params)
    return World(task, **world_params,
                 logging=False,
                 enable_visualization=enable_visualization)


def view_episode(episode):
    env = get_world(episode.task_name,
                    episode.task_params,
                    episode.world_params,
                    enable_visualization=True)
    env.reset()
    env.set_full_state(episode.initial_full_state)
    for time, observation, reward, action in zip(episode.timestamps,
                                                 episode.observations,
                                                 episode.rewards,
                                                 episode.robot_actions):
        env.step(action)
    env.close()


def view_policy(task, world_params, policy_fn, max_time_steps):
    env = get_world(task.task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=True)
    obs = env.reset()
    for time in range(max_time_steps):
        obs, reward, done, info = env.step(action=policy_fn(obs))
    env.close()


def record_video_of_policy(task, world_params, policy_fn, file_name, max_time_steps=100):
    env = get_world(task.task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=False)
    obs = env.reset()
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    recorder.capture_frame()
    for time in range(max_time_steps):
        obs, reward, done, info = env.step(action=policy_fn(obs))
        recorder.capture_frame()

    recorder
    env.close()


def record_video_of_episode(episode, file_name):
    env = get_world(episode.task_name,
                    episode.task_params,
                    episode.world_params,
                    enable_visualization=False)
    env.set_full_state(episode.initial_full_state)
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    recorder.capture_frame()
    for time, observation, reward, action in zip(episode.timestamps,
                                                 episode.observations,
                                                 episode.rewards,
                                                 episode.robot_actions):
        env.step(action)
        recorder.capture_frame()

    recorder.close()
    env.close()

