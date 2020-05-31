import numpy as np
import gym
import pybullet
import os
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.utils.env_utils import combine_spaces
from causal_rl_bench.task_generators.task import task_generator


class World(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, task=None, skip_frame=10,
                 enable_visualization=False, seed=0,
                 action_mode="joint_positions", observation_mode="structured",
                 normalize_actions=True, normalize_observations=True,
                 max_episode_length=None, data_recorder=None,
                 enable_goal_image=False, **kwargs):
        """
        Constructor sets up the physical world parameters,
        and resets to begin training.

        Args:
            skip_frame (int): the number of frames to skip for control
            enable_visualization (bool): if the simulation env is to be
                visualized
        """
        self.observation_mode = observation_mode
        self.enable_goal_image = enable_goal_image
        self.action_mode = action_mode
        self.seed(seed)
        self.simulation_time = 1./250
        self.robot = TriFingerRobot(action_mode=action_mode,
                                    observation_mode=observation_mode,
                                    enable_visualization=enable_visualization,
                                    skip_frame=skip_frame,
                                    normalize_actions=normalize_actions,
                                    normalize_observations=normalize_observations,
                                    enable_goal_image=enable_goal_image,
                                    simulation_time=self.simulation_time)
        self.pybullet_client = self.robot.get_pybullet_client()
        if enable_goal_image:
            self.stage = Stage(pybullet_client=self.pybullet_client,
                               observation_mode=observation_mode,
                               normalize_observations=normalize_observations,
                               goal_image_pybullet_instance=self.robot.
                               get_goal_image_instance_pybullet())
        else:
            self.stage = Stage(pybullet_client=self.pybullet_client,
                               observation_mode=observation_mode,
                               normalize_observations=normalize_observations)
        if max_episode_length is None:
            self.enforce_episode_length = False
        else:
            self.enforce_episode_length = True
        self.max_episode_length = max_episode_length
        self.episode_length = 0

        gym.Env.__init__(self)
        if task is None:
            self.task = task_generator("reaching")
        else:
            self.task = task

        self.task.init_task(self.robot, self.stage)
        self.reset_observations_space()
        self.action_space = self.robot.get_action_spaces()
        self.skip_frame = skip_frame
        self.dt = self.simulation_time * self.skip_frame
        self.metadata['video.frames_per_second'] = \
            (1 / self.simulation_time) / self.skip_frame
        # TODO: verify spaces here
        self._setup_viewing_camera()

        self.data_recorder = data_recorder
        self.tracker = Tracker(task=self.task,
                               world_params=self.get_world_params())
        self.scale_reward_by_dt = True
        self.reset()
        return

    def reset_observations_space(self):
        if not self.observation_mode == "cameras":
            self.robot.select_observations(self.task.task_robot_observation_keys)
            self.stage.select_observations(self.task.task_stage_observation_keys)
            self.observation_space = \
                combine_spaces(self.robot.get_observation_spaces(),
                               self.stage.get_observation_spaces())
        elif self.observation_mode == "cameras" and self.enable_goal_image:
            self.stage.select_observations(["goal_image"])
            self.observation_space = combine_spaces(
                self.robot.get_observation_spaces(),
                self.stage.get_observation_spaces())
        else:
            self.observation_space = self.robot.get_observation_spaces()
        return

    def step(self, action):
        self.episode_length += 1
        self.robot.apply_action(action)
        if self.observation_mode == "cameras" and self.enable_goal_image:
            current_images = self.robot.get_current_camera_observations()
            goal_images = self.stage.get_current_goal_image()
            observation = np.concatenate((current_images, goal_images),
                                         axis=0)
        elif self.observation_mode == "cameras":
            observation = self.robot.get_current_camera_observations()
        else:
            observation = self.task.filter_structured_observations()
        info = self.task.get_info()
        reward = self.task.get_reward()
        if self.scale_reward_by_dt:
            reward *= self.dt
        done = self._is_done()
        if self.data_recorder:
            self.data_recorder.append(robot_action=action,
                                      observation=observation,
                                      reward=reward,
                                      done=done,
                                      info=info,
                                      timestamp=self.episode_length *
                                                self.skip_frame *
                                                self.simulation_time)

        return observation, reward, done, info

    def sample_new_goal(self):
        return self.task.sample_new_goal()

    def add_data_recorder(self, data_recorder):
        self.data_recorder = data_recorder

    # def switch_task(self, task):
    #     self.task = task
    #     self.task.init_task(self.robot, self.stage)
    #     if not self.observation_mode == "cameras":
    #         self.robot.select_observations(self.task.task_robot_observation_keys)
    #         self.stage.select_observations(self.task.task_stage_observation_keys)
    #         self.observation_space = \
    #             combine_spaces(self.robot.get_observation_spaces(),
    #                            self.stage.get_observation_spaces())
    #     elif self.observation_mode == "cameras" and self.enable_goal_image:
    #         self.stage.select_observations(["goal_image"])
    #         self.observation_space = combine_spaces(
    #             self.robot.get_observation_spaces(),
    #             self.stage.get_observation_spaces())
    #     else:
    #         self.observation_space = self.robot.get_observation_spaces()
    #     self.action_space = self.robot.get_action_spaces()

    # def get_counterfactual_world(self):
    #     raise Exception(" ")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, interventions_dict=None):
        self.tracker.add_episode_experience(self.episode_length)
        self.episode_length = 0
        if interventions_dict is not None:
            self.tracker.do_intervention(self.task, interventions_dict)
        success_signal, interventions_info, reset_observation_space_signal = \
            self.task.reset_task(interventions_dict)
        if reset_observation_space_signal:
            self.reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self.tracker.add_invalid_intervention(interventions_info)
        # TODO: make sure that stage observations returned are up to date

        if self.data_recorder:
            self.data_recorder.new_episode(self.get_full_state(),
                                           task_name=self.task.task_name,
                                           task_params=self.task.get_task_params(),
                                           world_params=self.get_world_params())

        if self.observation_mode == "cameras" and self.enable_goal_image:
            current_images = self.robot.get_current_camera_observations()
            goal_images = self.stage.get_current_goal_image()
            return np.concatenate((current_images, goal_images), axis=0)
        elif self.observation_mode == "cameras":
            return self.robot.get_current_camera_observations()
        else:
            return self.task.filter_structured_observations()

    def close(self):
        if self.data_recorder:
            self.data_recorder.save()
        self.robot.close()

    def get_tracker(self):
        return self.tracker

    def enforce_max_episode_length(self, episode_length=2000):
        self.enforce_episode_length = True
        self.max_episode_length = episode_length

    def _is_done(self):
        if self.enforce_episode_length and \
                self.episode_length > self.max_episode_length:
            return True
        else:
            return self.task.is_done()

    def do_single_random_intervention(self):
        success_signal, interventions_info, interventions_dict, reset_observation_space_signal = \
            self.task.do_single_random_intervention()
        if reset_observation_space_signal:
            self.reset_observations_space()
        if len(interventions_dict) > 0:
            self.tracker.do_intervention(self.task, interventions_dict)
            if success_signal is not None:
                if not success_signal:
                    self.tracker.add_invalid_intervention(interventions_info)
        return interventions_dict

    def do_intervention(self, interventions_dict,
                        check_bounds=None):
        success_signal, interventions_info, reset_observation_space_signal = \
            self.task.do_intervention(interventions_dict,
                                      check_bounds=check_bounds)
        self.tracker.do_intervention(self.task, interventions_dict)
        if reset_observation_space_signal:
            self.reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self.tracker.add_invalid_intervention(interventions_info)
        return success_signal

    def get_full_state(self):
        full_state = []
        full_state.extend(self.robot.get_full_state())
        full_state.extend(self.stage.get_full_state())
        return np.array(full_state)

    def set_full_state(self, new_full_state):
        robot_state_size = self.robot.get_state_size()
        self.robot.set_full_state(new_full_state[0:robot_state_size])
        self.stage.set_full_state(new_full_state[robot_state_size:])
        return

    def render(self, mode="human"):
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(
            width=self._render_width, height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _setup_viewing_camera(self):
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -60
        self._render_width = 320
        self._render_height = 240
        base_pos = [0, 0, 0]
        self.view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)

    def get_current_task_parameters(self):
        return self.task.get_current_task_parameters()

    def get_world_params(self):
        world_params = dict()
        world_params["skip_frame"] = self.robot.get_skip_frame()
        world_params["action_mode"] = self.robot.get_action_mode()
        world_params["observation_mode"] = self.robot.get_observation_mode()
        world_params["normalize_actions"] = \
            self.robot.robot_actions.is_normalized()
        world_params["normalize_observations"] = \
            self.robot.robot_observations.is_normalized()
        world_params["max_episode_length"] = self.max_episode_length
        world_params["enable_goal_image"] = self.enable_goal_image
        world_params["simulation_time"] = self.simulation_time
        return world_params

    def save_world(self, log_relative_path):
        if not os.path.exists(log_relative_path):
            os.makedirs(log_relative_path)
        tracker_path = os.path.join(log_relative_path, 'tracker')
        tracker = self.get_tracker()
        tracker.save(file_path=tracker_path)
        return



