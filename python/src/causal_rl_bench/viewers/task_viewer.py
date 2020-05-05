from causal_rl_bench.envs.world import World
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
        frames = []
        image = env.render(mode="rgba_array")
        size = image.shape
        for time, world_state, reward, actions in zip(episode.timestamps,
                                                      episode.world_states,
                                                      episode.rewards,
                                                      episode.actions):
            env.set_full_state(world_state)
            image = env.render(mode="rgba_array")
            frames.append(image)
        output_path = os.path.join(self.path, task_params_dict["task_id"], "episode_{}.mp4".format(num))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=int(1. / task_params_dict["skip_frame"]), size=size)

        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

    def record_animation_batch_of_episodes(self, episodes):
        for num, episode in enumerate(episodes):
            self.record_animation_of_episode(episode, num)

    def view_episode(self, episode):
        task_params_dict = episode.task_params
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=True)
        for time, world_state, reward, actions in zip(episode.timestamps,
                                                      episode.world_states,
                                                      episode.rewards,
                                                      episode.actions):
            env.set_full_state(world_state)
        env.close()

    def view_batch_of_episodes(self, episodes):
        for episode in episodes:
            self.view_episode(episode)

    def record_animation_of_policy(self, task_params_dict, policy_wrapper, max_time_steps=100):
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=False)
        frames = []
        obs = env.reset()
        image = env.render(mode="rgba_array")
        size = image.shape
        for time in range(max_time_steps):
            env.step(action=policy_wrapper.get_action_for_observation(obs))
            image = env.render(mode="rgba_array")
            frames.append(image)
        output_path = os.path.join(self.path, task_params_dict["task_id"],
                                   "policy.mp4".format(policy_wrapper.get_identifier()))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=int(1. / task_params_dict["skip_frame"]), size=size)

        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

    def record_animation_batch_of_policies(self, task_params_dict,
                                           list_policy_wrappers,
                                           max_time_steps=100):
        for policy_wrapper in list_policy_wrappers:
            self.record_animation_of_policy(task_params_dict,
                                            policy_wrapper=policy_wrapper,
                                            max_time_steps=max_time_steps)

    def view_policy(self, task_params_dict, policy_wrapper, max_time_steps):
        env = get_world_for_task_parmas(task_params_dict, enable_visualization=True)
        obs = env.reset()
        for time in range(max_time_steps):
            obs = env.step(action=policy_wrapper.get_action_for_observation(obs))
        env.close()

    def view_batch_of_policies(self, task_params_dict, list_policy_wrappers, max_time_steps):
        for policy_wrapper in list_policy_wrappers:
            self.view_policy(task_params_dict,
                             policy_wrapper=policy_wrapper,
                             max_time_steps=max_time_steps)
