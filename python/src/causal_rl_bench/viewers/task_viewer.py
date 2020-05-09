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


class TaskViewer:
    def __init__(self, output_path=None):
        if output_path is None:
            self.path = os.path.join("output", "visualizations")
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        else:
            self.path = output_path

    def record_animation_of_episode(self, episode, num=0):
        env = get_world(episode.task_id,
                        episode.task_params,
                        episode.world_params,
                        enable_visualization=False)
        env.set_full_state(episode.initial_full_state)
        output_path = os.path.join(self.path, "{}_episode_{}.mp4".format(
            episode.world_params["task_id"], num))
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
        env = get_world(episode.task_id,
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

    def record_animation_of_policy(self, task, world_params, policy_wrapper, max_time_steps=100):
        env = get_world(task.name,
                        task.get_task_params(),
                        world_params,
                        enable_visualization=False)
        obs = env.reset()
        output_path = os.path.join(self.path, "{}_policy.mp4".format(world_params["task_id"]))
        recorder = VideoRecorder(env, output_path)
        recorder.capture_frame()
        for time in range(max_time_steps):
            obs = env.step(action=policy_wrapper.get_action_for_observation(obs))
            recorder.capture_frame()

        recorder.close()
        env.close()

    def view_policy(self, task, world_params, policy_wrapper, max_time_steps):
        env = get_world(task.name,
                        task.get_task_params(),
                        world_params,
                        enable_visualization=True)
        obs = env.reset()
        for time in range(max_time_steps):
            obs = env.step(action=policy_wrapper.get_action_for_observation(obs))
        env.close()

