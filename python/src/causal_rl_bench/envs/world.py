import numpy as np
import gym
import pybullet
import os
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.utils.env_utils import combine_spaces
from causal_rl_bench.task_generators.task import task_generator
import copy

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

        :param task:
        :param skip_frame:
        :param enable_visualization:
        :param seed:
        :param action_mode:
        :param observation_mode:
        :param normalize_actions:
        :param normalize_observations:
        :param max_episode_length:
        :param data_recorder:
        :param enable_goal_image:
        :param kwargs:
        """
        self.__observation_mode = observation_mode
        self.__enable_goal_image = enable_goal_image
        self.__action_mode = action_mode
        self.seed(seed)
        self.__simulation_time = 1. / 250
        self.__robot = TriFingerRobot(action_mode=action_mode,
                                      observation_mode=observation_mode,
                                      enable_visualization=enable_visualization,
                                      skip_frame=skip_frame,
                                      normalize_actions=normalize_actions,
                                      normalize_observations=normalize_observations,
                                      enable_goal_image=enable_goal_image,
                                      simulation_time=self.__simulation_time)
        self.__pybullet_client = self.__robot.get_pybullet_client()
        if enable_goal_image:
            self.__stage = Stage(pybullet_client=self.__pybullet_client,
                                 observation_mode=observation_mode,
                                 normalize_observations=normalize_observations,
                                 goal_image_pybullet_instance=self.__robot.
                                 get_goal_image_instance_pybullet())
        else:
            self.__stage = Stage(pybullet_client=self.__pybullet_client,
                                 observation_mode=observation_mode,
                                 normalize_observations=normalize_observations)

        gym.Env.__init__(self)
        if task is None:
            self.task = task_generator("reaching")
        else:
            self.task = task

        self.task.init_task(self.__robot, self.__stage)
        self.__reset_observations_space()
        self.action_space = self.__robot.get_action_spaces()
        self.__skip_frame = skip_frame
        self.dt = self.__simulation_time * self.__skip_frame
        self.metadata['video.frames_per_second'] = \
            (1 / self.__simulation_time) / self.__skip_frame
        # TODO: verify spaces here
        self.__setup_viewing_camera()

        if max_episode_length == np.inf:
            self.__enforce_episode_length = False
        elif max_episode_length is None:
            self.__enforce_episode_length = True
            max_episode_length = int(task.get_max_episode_length() / self.dt)
        else:
            self.__enforce_episode_length = True
        self.__max_episode_length = max_episode_length
        self.__episode_length = 0

        self.__data_recorder = data_recorder
        self.__wrappers_dict = dict()
        self.__tracker = Tracker(task=self.task,
                                 world_params=self._get_world_params())
        self.__scale_reward_by_dt = True
        self.__disabled_actions = False

        #TODO: I am not sure if this reset is necassary, TO BE CONFIRMED
        # self.reset()
        return

    def __reset_observations_space(self):
        """

        :return:
        """
        if not self.__observation_mode == "cameras":
            self.__robot.select_observations(self.task.task_robot_observation_keys)
            self.__stage.select_observations(self.task.task_stage_observation_keys)
            self.observation_space = \
                combine_spaces(self.__robot.get_observation_spaces(),
                               self.__stage.get_observation_spaces())
        elif self.__observation_mode == "cameras" and self.__enable_goal_image:
            self.__stage.select_observations(["goal_image"])
            self.observation_space = combine_spaces(
                self.__robot.get_observation_spaces(),
                self.__stage.get_observation_spaces())
        else:
            self.observation_space = self.__robot.get_observation_spaces()
        return

    def step(self, action):
        """

        :param action:
        :return:
        """
        self.__episode_length += 1
        if not self.__disabled_actions:
            self.__robot.apply_action(action)
        if self.__observation_mode == "cameras" and self.__enable_goal_image:
            current_images = self.__robot.get_current_camera_observations()
            goal_images = self.__stage.get_current_goal_image()
            observation = np.concatenate((current_images, goal_images),
                                         axis=0)
        elif self.__observation_mode == "cameras":
            observation = self.__robot.get_current_camera_observations()
        else:
            observation = self.task.filter_structured_observations()
        reward = self.task.get_reward()
        info = self.task.get_info()
        if self.__scale_reward_by_dt:
            reward *= self.dt
        done = self._is_done()
        if self.__data_recorder:
            self.__data_recorder.append(robot_action=action,
                                        observation=observation,
                                        reward=reward,
                                        done=done,
                                        info=info,
                                        timestamp=self.__episode_length *
                                                self.__skip_frame *
                                                self.__simulation_time)

        return observation, reward, done, info

    def reset_default_goal(self):
        """

        :return:
        """
        self.task.reset_default_state()
        return

    def sample_new_goal(self, training=True, level=None):
        """

        :param training:
        :param level:
        :return:
        """
        return self.task.sample_new_goal(training, level)

    def _disable_actions(self):
        self.__disabled_actions = True

    def _add_data_recorder(self, data_recorder):
        self.__data_recorder = data_recorder

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, interventions_dict=None):
        """

        :param interventions_dict:
        :return:
        """
        # self.__pybullet_client.resetSimulation()
        # optionally enable EGL for faster headless rendering
        # try:
        #     if os.environ["PYBULLET_EGL"]:
        #         con_mode = self._p.getConnectionInfo()['connectionMethod']
        #         if con_mode == self._p.DIRECT:
        #             egl = pkgutil.get_loader('eglRenderer')
        #             if (egl):
        #                 self._p.loadPlugin(egl.get_filename(),
        #                                    "_eglRendererPlugin")
        #             else:
        #                 self._p.loadPlugin("eglRendererPlugin")
        # except:
        #     pass
        self.__tracker.add_episode_experience(self.__episode_length)
        self.__episode_length = 0
        if interventions_dict is not None:
            interventions_dict = copy.deepcopy(interventions_dict)
            self.__tracker.do_intervention(self.task, interventions_dict)
        success_signal, interventions_info, reset_observation_space_signal = \
            self.task.reset_task(interventions_dict)
        if reset_observation_space_signal:
            self.__reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self.__tracker.add_invalid_intervention(interventions_info)
        # TODO: make sure that stage observations returned are up to date

        if self.__data_recorder:
            self.__data_recorder.new_episode(self.get_full_state(),
                                             task_name=self.task.task_name,
                                             task_params=self.task.get_task_params(),
                                             world_params=self._get_world_params())
        if self.__observation_mode == "cameras" and self.__enable_goal_image:
            current_images = self.__robot.get_current_camera_observations()
            goal_images = self.__stage.get_current_goal_image()
            return np.concatenate((current_images, goal_images), axis=0)
        elif self.__observation_mode == "cameras":
            return self.__robot.get_current_camera_observations()
        else:
            return self.task.filter_structured_observations()

    def close(self):
        """

        :return:
        """
        if self.__data_recorder:
            self.__data_recorder.save()
        self.__robot.close()

    def _get_tracker(self):
        return self.__tracker

    def enforce_max_episode_length(self, episode_length=2000):
        """

        :param episode_length:
        :return:
        """
        self.__enforce_episode_length = True
        self.__max_episode_length = episode_length

    def _is_done(self):
        if self.__enforce_episode_length and \
                self.__episode_length > self.__max_episode_length:
            return True
        else:
            return self.task.is_done()

    def do_single_random_intervention(self):
        """

        :return:
        """
        success_signal, interventions_info, interventions_dict, reset_observation_space_signal = \
            self.task.do_single_random_intervention()
        if reset_observation_space_signal:
            self.__reset_observations_space()
        if len(interventions_dict) > 0:
            self.__tracker.do_intervention(self.task, interventions_dict)
            if success_signal is not None:
                if not success_signal:
                    self.__tracker.add_invalid_intervention(interventions_info)
        return interventions_dict

    def do_intervention(self, interventions_dict,
                        check_bounds=None):
        """

        :param interventions_dict:
        :param check_bounds:
        :return:
        """
        success_signal, interventions_info, reset_observation_space_signal = \
            self.task.do_intervention(interventions_dict,
                                      check_bounds=check_bounds)
        self.__tracker.do_intervention(self.task, interventions_dict)
        if reset_observation_space_signal:
            self.__reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self.__tracker.add_invalid_intervention(interventions_info)
        return success_signal

    def get_full_state(self):
        """

        :return:
        """
        full_state = []
        full_state.extend(self.__robot.get_full_state())
        full_state.extend(self.__stage.get_full_state())
        return np.array(full_state)

    def set_full_state(self, new_full_state):
        """

        :param new_full_state:
        :return:
        """
        robot_state_size = self.__robot.get_state_size()
        self.__robot.set_full_state(new_full_state[0:robot_state_size])
        self.__stage.set_full_state(new_full_state[robot_state_size:])
        return

    def render(self, mode="human"):
        """

        :param mode:
        :return:
        """
        (_, _, px, _, _) = self.__pybullet_client.getCameraImage(
            width=self._render_width, height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def __setup_viewing_camera(self):
        """

        :return:
        """
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -60
        self._render_width = 320
        self._render_height = 240
        base_pos = [0, 0, 0]
        self.view_matrix = self.__pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = self.__pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)

    def get_current_task_parameters(self):
        """

        :return:
        """
        return self.task.get_current_task_parameters()

    def _get_world_params(self):
        """

        :return:
        """
        world_params = dict()
        world_params["skip_frame"] = self.__robot.get_skip_frame()
        world_params["action_mode"] = self.__robot.get_action_mode()
        world_params["observation_mode"] = self.__robot.get_observation_mode()
        world_params["normalize_actions"] = \
            self.__robot.robot_actions.is_normalized()
        world_params["normalize_observations"] = \
            self.__robot.robot_observations.is_normalized()
        world_params["max_episode_length"] = self.__max_episode_length
        world_params["enable_goal_image"] = self.__enable_goal_image
        world_params["simulation_time"] = self.__simulation_time
        world_params["wrappers"] = self.__wrappers_dict
        return world_params

    def _add_wrapper_info(self, wrapper_dict):
        """

        :param wrapper_dict:
        :return:
        """
        self.__wrappers_dict.update(wrapper_dict)
        return

    def save_world(self, log_relative_path):
        """

        :param log_relative_path:
        :return:
        """
        if not os.path.exists(log_relative_path):
            os.makedirs(log_relative_path)
        tracker_path = os.path.join(log_relative_path, 'tracker')
        tracker = self._get_tracker()
        tracker.save(file_path=tracker_path)
        return

    def is_in_training_mode(self):
        """

        :return:
        """
        return self.task.is_in_training_mode()

    def get_joint_positions_lower_bound(self):
        return self.__robot.robot_actions.\
            joint_positions_lower_bounds

    def get_action_mode(self):
        return self.__action_mode

    def set_action_mode(self, action_mode):
        self.__action_mode = action_mode
        self.__robot.set_action_mode(action_mode)

    def get_robot(self):
        return self.__robot

    def get_stage(self):
        return self.__stage

    def get_tracker(self):
        return self.__tracker
