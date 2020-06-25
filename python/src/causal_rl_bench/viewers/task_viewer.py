from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_rl_bench.task_generators.task import task_generator
import numpy as np


def get_world(task_generator_id, task_params, world_params,
              enable_visualization=False, env_wrappers=np.array([]),
              env_wrappers_args=np.array([])):
    world_params["skip_frame"] = 1
    if task_params is None:
        task = task_generator(task_generator_id)
    else:
        task = task_generator(task_generator_id, **task_params)
    env = World(task, **world_params, enable_visualization=enable_visualization)
    for i in range(len(env_wrappers)):
        env = env_wrappers[i](env, **env_wrappers_args[i])
    return env


def view_episode(episode, env_wrappers=np.array([]),
                 env_wrappers_args=np.array([])):
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode._task_name,
                    episode._task_params,
                    episode.world_params,
                    enable_visualization=True,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
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
                number_of_resets, env_wrappers=np.array([]),
                env_wrappers_args=np.array([])):
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task._task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=True,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    for reset_idx in range(number_of_resets):
        obs = env.reset()
        for time in range(int(max_time_steps/number_of_resets)):
            #compute next action
            desired_action = policy_fn(obs)
            for _ in range(actual_skip_frame):
                obs, reward, done, info = env.step(action=desired_action)
    env.close()


def record_video_of_policy(task, world_params, policy_fn, file_name,
                           number_of_resets, max_time_steps=100,
                           env_wrappers=np.array([]),
                           env_wrappers_args=np.array([])):
    #TODO: discuss the speed of the current render method since it takes a long time to render a frame
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task._task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=False,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    for reset_idx in range(number_of_resets):
        obs = env.reset()
        recorder.capture_frame()
        for i in range(max_time_steps):
            desired_action = policy_fn(obs)
            for _ in range(actual_skip_frame):
                obs, reward, done, info = env.step(action=desired_action)
                recorder.capture_frame()
    env.close()


def record_video_of_random_policy(task, world_params, file_name,
                                  number_of_resets, max_time_steps=100,
                                  env_wrappers=np.array([]),
                                  env_wrappers_args=np.array([])):
    #TODO: discuss the speed of the current render method since it takes a
    # long time to render a frame
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task._task_name,
                    task.get_task_params(),
                    world_params,
                    enable_visualization=False,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    for reset_idx in range(number_of_resets):
        obs = env.reset()
        recorder.capture_frame()
        for i in range(max_time_steps):
            for _ in range(actual_skip_frame):
                obs, reward, done, info = \
                    env.step(action=env.action_space.sample())
                recorder.capture_frame()
    env.close()


def record_video_of_episode(episode, file_name, env_wrappers=np.array([]),
                            env_wrappers_args=np.array([])):
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode._task_name,
                    episode._task_params,
                    episode.world_params,
                    enable_visualization=False,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
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

