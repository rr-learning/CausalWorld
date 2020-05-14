from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_rl_bench.tasks.task import Task
import time


def get_world(task_id, task_params, world_params, enable_visualization=False):
    world_params["skip_frame"] = 1
    if task_params is None:
        task = Task(task_id)
    else:
        task = Task(task_id, **task_params)
    return World(task, **world_params,
                 logging=False,
                 enable_visualization=enable_visualization)


def view_episode(episode):
    actual_skip_frame = episode.world_params["skip_frame"]
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
        for _ in range(actual_skip_frame):
            env.step(action)
    env.close()


def view_policy(task, world_params, policy_fn, max_time_steps,
                number_of_resets):
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task.task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=True)
    for reset_idx in range(number_of_resets):
        obs = env.reset()
        for time in range(int(max_time_steps/number_of_resets)):
            for _ in range(actual_skip_frame):
                obs, reward, done, info = env.step(action=policy_fn(obs))
    env.close()


def record_video_of_policy(task, world_params, policy_fn, file_name, max_time_steps=100):
    #TODO: discuss the speed of the current render method since it takes a long time to render a frame
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task.task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=False)
    obs = env.reset()
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    recorder.capture_frame()
    for i in range(max_time_steps):
        print(i)
        for _ in range(actual_skip_frame):
            start = time.time()
            obs, reward, done, info = env.step(action=policy_fn(obs))
            recorder.capture_frame()
            end = time.time()
            print(end-start)

    env.close()


def record_video_of_episode(episode, file_name):
    actual_skip_frame = episode.world_params["skip_frame"]
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
        for _ in range(actual_skip_frame):
            env.step(action)
            recorder.capture_frame()

    recorder.close()
    env.close()

