from counterfactual.python.src.causal_rl_bench.envs.base import Base
from counterfactual.python.src.causal_rl_bench.envs.scene.objects import Cuboid
from counterfactual.python.src.causal_rl_bench.envs.scene.silhouette import SCuboid
from counterfactual.python.src.causal_rl_bench.envs.robot.observations import (SingleCameraObservation,
                                                                               AllCameraObservation)
import numpy as np
import gym
import math
import matplotlib.pyplot as plt

from pybullet_fingers import sample as pf_sample  # TODO: Add own sampling utils


class CSWrapper(gym.Wrapper):
    def __init__(self, env, silhouette_size=None, reward_mode="volume_fraction", unit_length=0.065):
        super().__init__(env)
        if silhouette_size is None:
            silhouette_size = [2, 3, 1]
        self.silhouette_size = np.array(silhouette_size)
        self.silhouette_position = np.array([0, 0, 0.0115 + silhouette_size[2] / 2 * unit_length])
        self.silhouette_orientation = [0, 0, 0, 1]
        self.silhouette_subgoals = []
        start_position = self.silhouette_position - (self.silhouette_size - 1) / 2 * unit_length

        for n_x in range(silhouette_size[0]):
            for n_y in range(silhouette_size[1]):
                for n_z in range(silhouette_size[2]):
                    # Add sub goals as invisible silhouettes
                    s_position = start_position + np.array([n_x, n_y, n_z]) * unit_length
                    s_silhouette = SCuboid(size=[unit_length]*3,
                                           position=s_position,
                                           orientation=self.silhouette_orientation,
                                           alpha=0.1)
                    self.silhouette_subgoals.append(s_silhouette)

        self.silhouette_volume = np.prod(self.silhouette_size) * math.pow(unit_length, 3)
        self.silhouette = SCuboid(size=self.silhouette_size * unit_length,
                                  position=list(self.silhouette_position),
                                  orientation=self.silhouette_orientation)
        self.env.add_scene_object(self.silhouette)
        self.observation_space = self.env.observation_space  # TODO: This seems to be not properly done
        self.reward_mode = reward_mode

    def _compute_silhouette_reward(self):
        filled_volume = 0.0
        sil_reward = 0.0
        sil_done = False
        for scene_object in self.env.scene_objects:
            if type(scene_object) == Cuboid:
                # Not sure yet how we can calculate how much volume of the scene_object fills the silhouette
                pass
        filled_volume_fraction = filled_volume / self.silhouette_volume

        if filled_volume_fraction > 0.95:
            done = True

        if self.reward_mode == 'dense':
            pass
        elif self.reward_mode == "volume_fraction":
            sil_reward = filled_volume_fraction
        elif self.reward_mode == "sparse":
            if sil_done:
                sil_reward = 1.0

        return sil_reward, sil_done

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        silhouette_rew, silhouette_done = self._compute_silhouette_reward()
        rew += silhouette_rew
        done = silhouette_done
        return obs, rew, done, info


def make_env(horizon=1000, enable_visualisation=True, control_rate=0.02, seed=0,
             observation_mode="structured",
             silhouette_size=None, reward_mode="volume_fraction",
             action_mode="joint_positions", unit_length=0.065):

    if silhouette_size is None:
        silhouette_size = [2, 3, 1]

    env = Base(control_rate, enable_visualisation, seed, action_mode)
    env.enforce_max_episode_length(horizon)

    number_of_blocks = int(np.prod(silhouette_size))
    for i in range(number_of_blocks):
        min_angle = i / number_of_blocks * 2 * math.pi
        max_angle = (i + 1) / number_of_blocks * 2 * math.pi
        block_position = pf_sample.random_position_in_arena(height_limits=0.0115 + unit_length / 2,
                                                            radius_limits=(0.05, 0.13),
                                                            angle_limits=(min_angle, max_angle))
        cuboid = Cuboid(size=[unit_length]*3,
                        position=block_position)
        env.add_scene_object(cuboid)

    # TODO: Which resolution do we set or can this be chosen by the user?
    if observation_mode == "all_cameras":
        env = AllCameraObservation(env)
    elif observation_mode == "single_camera":
        env = SingleCameraObservation(env)

    # Add the Wrapper to manage and display the Silhouette
    env = CSWrapper(env, silhouette_size, reward_mode, unit_length)

    return env


if __name__ == "__main__":
    # For now, this is just to check the functioning of the environment tasks for now
    env = make_env(horizon=1000, enable_visualisation=True, control_rate=0.02,
                   seed=0, observation_mode="all_cameras", silhouette_size=[1, 2, 3],
                   reward_mode="volume_fraction", unit_length=0.04)
    env.activate_camera_observations()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
