from causal_world.envs.causalworld import CausalWorld
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_world.task_generators.task import generate_task
import numpy as np


def view_episode(episode,
                 env_wrappers=np.array([]),
                 env_wrappers_args=np.array([])):
    """
    Visualizes a logged episode in the GUI

    :param episode: (Episode) the logged episode
    :param env_wrappers: (list) a list of gym wrappers
    :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
    :return:
    """
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode.task_name,
                    episode.task_params,
                    episode.world_params,
                    enable_visualization=True,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    env.reset()
    env.set_starting_state(episode.initial_full_state, check_bounds=False)
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
    Visualizes a policy for a specified environment in the GUI

    :param task: (Task) the task of the environment
    :param world_params: (dict) the world_params of the environment
    :param policy_fn: the policy to be evaluated
    :param max_time_steps: (int) the maximum number of time steps per episode
    :param number_of_resets: (int) the number of resets/episodes to be viewed
    :param env_wrappers: (list) a list of gym wrappers
    :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
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
    Records a video of a policy for a specified environment

    :param task: (Task) the task of the environment
    :param world_params: (dict) the world_params of the environment
    :param policy_fn: the policy to be evaluated
    :param file_name: (str) full path where the video is being stored.
    :param number_of_resets: (int) the number of resets/episodes to be viewed
    :param max_time_steps: (int) the maximum number of time steps per episode
    :param env_wrappers: (list) a list of gym wrappers
    :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
    :return:
    """
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
    Records a video of a random policy for a specified environment

    :param task: (Task) the task of the environment
    :param world_params: (dict) the world_params of the environment
    :param file_name: (str) full path where the video is being stored.
    :param number_of_resets: (int) the number of resets/episodes to be viewed
    :param max_time_steps: (int) the maximum number of time steps per episode
    :param env_wrappers: (list) a list of gym wrappers
    :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
    :return:
    """

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
     Records a video of a logged episode for a specified environment

     :param episode: (Episode) the logged episode
     :param file_name: (str) full path where the video is being stored.
     :param env_wrappers: (list) a list of gym wrappers
     :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
     :return:
     """
    actual_skip_frame = episode.world_params["skip_frame"]
    env = get_world(episode.task_name,
                    episode.task_params,
                    episode.world_params,
                    enable_visualization=False,
                    env_wrappers=env_wrappers,
                    env_wrappers_args=env_wrappers_args)
    env.set_starting_state(episode.initial_full_state, check_bounds=False)
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


def get_world(task_generator_id,
              task_params,
              world_params,
              enable_visualization=False,
              env_wrappers=np.array([]),
              env_wrappers_args=np.array([])):
    """
    Returns a particular CausalWorld instance with optional wrappers

    :param task_generator_id: (str) id of the task of the environment
    :param task_params: (dict) task params of the environment
    :param world_params: (dict) world_params of the environment
    :param enable_visualization: (bool) if GUI visualization is enabled
    :param env_wrappers: (list) a list of gym wrappers
    :param env_wrappers_args: (list) a list of kwargs for the gym wrappers
    :return: (CausalWorld) a CausalWorld environment instance
    """
    world_params["skip_frame"] = 1
    if task_params is None:
        task = generate_task(task_generator_id)
    else:
        if "task_name" in task_params:
            del task_params["task_name"]
        task = generate_task(task_generator_id, **task_params)
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


def record_video(env,
                 policy,
                 file_name,
                 number_of_resets=1,
                 max_time_steps=None):
    """
    Records a video of a policy for a specified environment
    :param env: (causal_world.CausalWorld) the environment to use for
                                           recording.
    :param policy: the policy to be evaluated
    :param file_name: (str) full path where the video is being stored.
    :param number_of_resets: (int) the number of resets/episodes to be viewed
    :param max_time_steps: (int) the maximum number of time steps per episode
    :return:
    """
    recorder = VideoRecorder(env, "{}.mp4".format(file_name))
    for reset_idx in range(number_of_resets):
        policy.reset()
        obs = env.reset()
        recorder.capture_frame()
        if max_time_steps is not None:
            for i in range(max_time_steps):
                desired_action = policy.act(obs)
                obs, reward, done, info = env.step(action=desired_action)
                recorder.capture_frame()
        else:
            while True:
                desired_action = policy.act(obs)
                obs, reward, done, info = env.step(action=desired_action)
                recorder.capture_frame()
                if done:
                    break
    recorder.close()
    return
