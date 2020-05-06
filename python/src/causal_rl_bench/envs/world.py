import numpy as np
import gym
import pybullet
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.cuboid_silhouette import CuboidSilhouette
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.envs.env_utils import combine_spaces


class World(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, task=None, task_id="picking", skip_frame=20,
                 enable_visualization=True, seed=0,
                 action_mode="joint_positions", observation_mode="structured",
                 camera_skip_frame=40, normalize_actions=True,
                 normalize_observations=True, max_episode_length=None,
                 data_recorder=None, **kwargs):
        """
        Constructor sets up the physical world parameters,
        and resets to begin training.

        Args:
            skip_frame (int): the number of frames to skip for control
            enable_visualization (bool): if the simulation env is to be
                visualized
        """
        self.robot = TriFingerRobot(action_mode=action_mode,
                                    observation_mode=observation_mode,
                                    enable_visualization=enable_visualization,
                                    skip_frame=skip_frame,
                                    camera_skip_frame=camera_skip_frame,
                                    normalize_actions=normalize_actions,
                                    normalize_observations=normalize_observations)
        self.seed(seed)
        self.stage = Stage(observation_mode=observation_mode,
                           normalize_observations=normalize_observations)
        if max_episode_length is None:
            self.enforce_episode_length = False
        else:
            self.enforce_episode_length = True
        self.max_episode_length = max_episode_length
        self.episode_length = 0
        self.simulation_time = 0.001
        gym.Env.__init__(self)
        if task is None:
            if task_id == "picking":
                self.task = PickingTask()
            elif task_id == "pushing":
                self.task = PushingTask()
            elif task_id == "cuboid_silhouette":
                self.task = CuboidSilhouette()
        else:
            self.task = task
        self.task.init_task(self.robot, self.stage)
        selected_observations = self.task.observation_keys
        self.robot.select_observations(selected_observations)
        self.stage.select_observations(selected_observations)
        self.observation_space = \
            combine_spaces(self.robot.get_observation_spaces(),
                           self.stage.get_observation_spaces())
        self.action_space = self.robot.get_action_spaces()
        self.data_recorder = data_recorder
        self.skip_frame = skip_frame
        self.metadata['video.frames_per_second'] = \
            (1 / self.simulation_time) / self.skip_frame
        #TODO: verify spaces here
        self.max_time_steps = 5000

        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -60
        self._render_width = 640
        self._render_height = 480
        self._p = self.robot.get_pybullet_client()
        base_pos = [0, 0, 0]
        self.view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = self._p.computeProjectionMatrixFOV(
                      fov=60, aspect=float(self._render_width) / self._render_height,
                      nearVal=0.1, farVal=100.0)
        self.reset()
        return

    def step(self, action):
        self.episode_length += 1

        self.robot.apply_action(action)
        task_observations = self.task.filter_observations()
        reward = self.task.get_reward()
        done = self._is_done()
        info = {}

        if self.data_recorder:
            self.data_recorder.append(robot_action=action,
                                      observation=task_observations,
                                      reward=reward,
                                      timestamp=self.episode_length *
                                                self.skip_frame *
                                                self.simulation_time)

        return task_observations, reward, done, info

    def sample_new_task(self):
        raise Exception(" ")

    def switch_task(self, task):
        self.task = task
        self.task.init_task(self.robot, self.stage)
        selected_observations = self.task.observation_keys
        self.robot.select_observations(selected_observations)
        self.stage.select_observations(selected_observations)
        self.observation_space = \
            combine_spaces(self.robot.get_observation_spaces(),
                           self.stage.get_observation_spaces())
        self.action_space = self.robot.get_action_spaces()

    def get_counterfactual_world(self):
        raise Exception(" ")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        self.episode_length = 0
        self.task.reset_task()
        #TODO: make sure that stage observations returned are up to date

        if self.data_recorder:
            self.data_recorder.new_episode(self.get_full_state(), task_params=self.task.get_task_params())
        return self.task.filter_observations()

    def close(self):
        if self.data_recorder:
            self.data_recorder.save()
        self.robot.close()

    def enforce_max_episode_length(self, episode_length=2000):
        self.enforce_episode_length = True
        self.max_episode_length = episode_length

    def _is_done(self):
        if self.enforce_episode_length and self.episode_length > self.max_episode_length:
            return True
        else:
            return self.task.is_done()

    def do_random_intervention(self):
        self.task.do_random_intervention()

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
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
