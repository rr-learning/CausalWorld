from causal_world.envs.causalworld import CausalWorld
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_world.task_generators.task import task_generator
import numpy as np


def get_world(task_generator_id,
              task_params,
              world_params,
              enable_visualization=False,
              env_wrappers=np.array([]),
              env_wrappers_args=np.array([])):
    """

    :param task_generator_id:
    :param task_params:
    :param world_params:
    :param enable_visualization:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    world_params["skip_frame"] = 1
    if task_params is None:
        task = task_generator(task_generator_id)
    else:
        del task_params["task_name"]
        task = task_generator(task_generator_id, **task_params)
    if "enable_visualization" in world_params.keys():
        world_params_temp = dict(world_params)
        del world_params_temp["enable_visualization"]
        env = CausalWorld(task,
                          **world_params_temp,
                          enable_visualization=enable_visualization)
    else:
        env = CausalWorld(task,
                          **world_params,
                          enable_visualization=enable_visualization)
    for i in range(len(env_wrappers)):
        env = env_wrappers[i](env, **env_wrappers_args[i])
    return env


def view_episode(episode,
                 env_wrappers=np.array([]),
                 env_wrappers_args=np.array([])):
    """

    :param episode:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode.get_task_name(),
                    episode.task_params,
                    episode.world_params,
                    enable_visualization=True,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    env.reset()
    env.set_starting_state(episode.initial_full_state)
    env.reset()
    for time, observation, reward, action in zip(episode.timestamps,
                                                 episode.observations,
                                                 episode.rewards,
                                                 episode.robot_actions):
        for _ in range(actual_skip_frame):
            env.step(action)
    env.close()


def view_policy(task,
                world_params,
                policy_fn,
                max_time_steps,
                number_of_resets,
                env_wrappers=np.array([]),
                env_wrappers_args=np.array([])):
    """

    :param task:
    :param world_params:
    :param policy_fn:
    :param max_time_steps:
    :param number_of_resets:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task.get_task_name(),
                    task.get_task_params(),
                    world_params,
                    enable_visualization=True,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    for reset_idx in range(number_of_resets):
        obs = env.reset()
        for time in range(int(max_time_steps / number_of_resets)):
            #compute next action
            desired_action = policy_fn(obs)
            for _ in range(actual_skip_frame):
                obs, reward, done, info = env.step(action=desired_action)
    env.close()


def record_video_of_policy(task,
                           world_params,
                           policy_fn,
                           file_name,
                           number_of_resets,
                           max_time_steps=100,
                           env_wrappers=np.array([]),
                           env_wrappers_args=np.array([])):
    """

    :param task:
    :param world_params:
    :param policy_fn:
    :param file_name:
    :param number_of_resets:
    :param max_time_steps:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    #TODO: discuss the speed of the current render method since it takes a long time to render a frame
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task.get_task_name(),
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
    recorder.close()
    env.close()


def record_video_of_random_policy(task,
                                  world_params,
                                  file_name,
                                  number_of_resets,
                                  max_time_steps=100,
                                  env_wrappers=np.array([]),
                                  env_wrappers_args=np.array([])):
    """

    :param task:
    :param world_params:
    :param file_name:
    :param number_of_resets:
    :param max_time_steps:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    #TODO: discuss the speed of the current render method since it takes a
    # long time to render a frame
    actual_skip_frame = world_params["skip_frame"]
    env = get_world(task.get_task_name(),
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
    recorder.close()
    env.close()


def record_video_of_episode(episode,
                            file_name,
                            env_wrappers=np.array([]),
                            env_wrappers_args=np.array([])):
    """

    :param episode:
    :param file_name:
    :param env_wrappers:
    :param env_wrappers_args:
    :return:
    """
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode.get_task_name(),
                    episode._task_params,
                    episode.world_params,
                    enable_visualization=False,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    env.set_starting_state(episode.initial_full_state)
    env.reset()
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