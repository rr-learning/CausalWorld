from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import cv2
import os


def get_world_for_task_parmas(task_params_dict, enable_visualization=False):
    return World(task_id=task_params_dict["task_id"],
                 skip_frame=task_params_dict["skip_frame"],
                 enable_visualization=enable_visualization,
                 action_mode=task_params_dict["action_mode"],
                 observation_mode=task_params_dict["observation_mode"],
                 camera_skip_frame=task_params_dict["camera_skip_frame"],
                 normalize_actions=task_params_dict["normalize_actions"],
                 normalize_observations=task_params_dict["normalize_observations"],
                 max_episode_length=task_params_dict["max_episode_length"],
                 logging=False)


class TaskViewer:
    def __init__(self, output_path=None):
        if output_path is None:
            self.path = os.path.join("output", "visualizations")
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        else:
            self.path = output_path

    def record_animation_of_episode(self, episode, num=0):
        task_params_dict = episode.task_params
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=False)
        env.set_full_state(episode.initial_full_state)
        output_path = os.path.join(self.path, "{}_episode_{}.mp4".format(task_params_dict["task_id"], num))
        recorder = VideoRecorder(env, output_path)
        recorder.capture_frame()
        for time, observation, reward, action in zip(episode.timestamps,
                                                     episode.observations,
                                                     episode.rewards,
                                                     episode.robot_actions):
            env.step(action)
            recorder.capture_frame()

        recorder.close()
        env.close()

    def view_episode(self, episode):
        task_params_dict = episode.task_params
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=True)
        env.reset()
        env.set_full_state(episode.initial_full_state)
        for time, observation, reward, action in zip(episode.timestamps,
                                                     episode.observations,
                                                     episode.rewards,
                                                     episode.robot_actions):
            env.step(action)
        env.close()

    def record_animation_of_policy(self, task_params_dict, policy_wrapper, max_time_steps=100):
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=False)
        obs = env.reset()
        output_path = os.path.join(self.path, "{}_policy.mp4".format(task_params_dict["task_id"]))
        recorder = VideoRecorder(env, output_path)
        recorder.capture_frame()
        for time in range(max_time_steps):
            obs = env.step(action=policy_wrapper.get_action_for_observation(obs))
            recorder.capture_frame()

        recorder.close()
        env.close()

    def view_policy(self, task_params_dict, policy_wrapper, max_time_steps):
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=True)
        obs = env.reset()
        for time in range(max_time_steps):
            obs = env.step(action=policy_wrapper.get_action_for_observation(obs))
        env.close()

