import numpy as np
import time
import pybullet
import gym

from pybullet_fingers.sim_finger import SimFinger
from pybullet_fingers import sample as pf_sample

from counterfactual.python.src.causal_rl_bench.envs import env_utils
from counterfactual.python.src.causal_rl_bench.envs.robot.action import TriFingerActionSpace
from counterfactual.python.src.causal_rl_bench.envs.robot.observations import TriFingerStructuredObservationSpaces


class Base(gym.Env):
    """
    Base environment of the robot manipulation task
    """

    def __init__(
        self, control_rate_s, enable_visualization, seed=0, action_mode="joint_position"
    ):
        """
        Constructor sets up the physical world parameters,
        and resets to begin training.

        Args:
            control_rate_s (float): the rate at which the env step runs
            finger-type (str "single"/"tri"): to train on the "single"
                or the "tri" finger
            enable_visualization (bool): if the simulation env is to be
                visualized
        """

        gym.Env.__init__(self)

        # Base contains the logger of the episode
        # self.logger = loggers.DataLogger()
        # TODO: add logger class

        self.action_mode = action_mode
        self.num_fingers = 3  # TODO: get rid of this parameter, or think about it
        simulation_rate_s = 0.001
        self.camera_rate_s = 0.3
        self.camera_skip_steps = int(round(self.camera_rate_s / control_rate_s))
        self.steps_per_control = int(round(control_rate_s / simulation_rate_s))
        assert (
            abs(control_rate_s - self.steps_per_control * simulation_rate_s)
            <= 0.000001
        )
        assert (
                abs(
                    self.camera_rate_s - self.camera_skip_steps * control_rate_s)
                <= 0.000001
        )

        self.seed(seed)

        # Base contains by default an observation space that represents the ground
        # truth states of the fingers and the scene_objects in the stage
        self.observations_keys = [
            "joint_positions",
            "joint_velocities",
            "action_joint_positions",
            "end_effector_positions"
        ]
        self.observations_sizes = [
            3 * self.num_fingers,
            3 * self.num_fingers,
            3 * self.num_fingers,
            3 * self.num_fingers
        ]

        self.robot_action_space = TriFingerActionSpace(action_mode=self.action_mode)
        self.spaces = TriFingerStructuredObservationSpaces(self.observations_keys, self.observations_sizes)

        self.finger = SimFinger(time_step=simulation_rate_s,
                                enable_visualization=enable_visualization,
                                finger_type="tri")

        self.enforce_episode_length = False

        self.metadata = {"render.modes": ["human", "rgb_array"]}

        # TODO: Not yet super happy with having a scaled and unscaled version for each space
        self.unscaled_observation_space = self.spaces.get_unscaled_observation_space()
        self.unscaled_action_space = self.robot_action_space.get_unscaled_action_space()

        self.observation_space = self.spaces.get_scaled_observation_space()
        self.action_space = self.robot_action_space.get_scaled_action_space()

        # List of all the objects in the scene
        self.scene_objects = SceneObjects(spaces)

        self.latest_observation = None
        self.max_time_steps = 5000
        self.control_index = -1
        self.camera_obs = False
        self.reset()

    def add_scene_object(self, scene_object):
        object_no = len(self.scene_objects)
        self.scene_objects.append(scene_object)
        so_observation_keys = ["object_position_{}".format(object_no),
                               "object_orientation_{}".format(object_no),
                               "object_id_{}".format(object_no)]
        so_observation_sizes = [3, 4, 1]

        self.observations_keys.extend(so_observation_keys)
        self.observations_sizes.extend(so_observation_sizes)

        self.spaces.add_scene_object(so_observation_keys,
                                     so_observation_sizes)
        self.unscaled_observation_space = self.spaces.get_unscaled_observation_space()
        self.observation_space = self.spaces.get_scaled_observation_space()

    def activate_camera_observations(self):
        self.camera_obs = True

    def _get_state(self, observation, action):
        """
        Get the current observation from the env for the agent

        Args:

        Returns:
            observation (list): comprising of the observations corresponding
                to the key values in the observation_keys
        """
        joint_positions = observation.position
        joint_velocities = observation.velocity
        tip_positions = self.finger.pinocchio_utils.forward_kinematics(
            joint_positions
        )
        end_effector_position = np.concatenate(tip_positions)

        observation_dict = {}
        observation_dict["joint_positions"] = joint_positions
        observation_dict["joint_velocities"] = joint_velocities
        observation_dict["end_effector_positions"] = end_effector_position
        observation_dict["action_joint_positions"] = action

        for object_no, scene_object in enumerate(self.scene_objects):
            observation_dict["object_position_{}".format(object_no)], observation_dict["object_orientation_{}".format(object_no)] = scene_object.get_state()
            observation_dict["object_id_{}".format(object_no)] = [float(object_no)]

        observation = [
            v
            for key in self.spaces.observations_keys
            for v in observation_dict[key]
        ]

        return observation

    def step(self, action):
        """
        The env step method

        Args:
            action (list): the joint positions that have to be achieved

        Returns:
            the observation scaled to lie between [-1;1], the reward,
            the done signal, and info on if the agent was successful at
            the current step
        """
        self.control_index += 1
        unscaled_action = env_utils.unscale(action, self.unscaled_action_space)
        if self.action_mode == "joint_positions":
            finger_action = self.finger.Action(position=unscaled_action)
        elif self.action_mode == "torques":
            finger_action = self.finger.Action(torque=unscaled_action)
        elif self.action_mode == "both":
            finger_action = self.finger.Action(torque=unscaled_action[:9], position=unscaled_action[9:])
        else:
            finger_action = self.finger.Action(position=unscaled_action)

        for _ in range(self.steps_per_control):
            t = self.finger.append_desired_action(finger_action)
            self.finger.step_simulation()
        if self.camera_obs and self.control_index % self.camera_skip_steps == 0:
            observation = self.finger.get_observation(t, update_images=True)
        else:
            observation = self.finger.get_observation(t, update_images=False)
        self.latest_observation = observation
        state = self._get_state(observation, unscaled_action)
        reward, done = self._is_done()
        info = {"is_success": np.float32(done)}

        scaled_observation = env_utils.scale(
            state, self.unscaled_observation_space
        )
        return scaled_observation, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Episode reset

        Returns:
            the scaled to [-1;1] observation from the env after the reset
        """
        # TODO: replace this with benchmark specific sampling methods, i.e. only at request of user
        action = pf_sample.feasible_random_joint_positions_for_reaching(
            self.finger, self.robot_action_space.action_bounds
        )
        observation = self.finger.reset_finger(action)

        for scene_object in self.scene_objects:
            if scene_object.is_not_fixed():
                pass
                # scene_object.sample_random_state_in_arena()
                # TODO: add functionality that scene_objects do not collide

        self.latest_observation = observation
        self.control_index = -1
        return env_utils.scale(
            self._get_state(observation, action),
            self.unscaled_observation_space,
        )


    def close(self):
        self.finger.disconnect_from_simulation()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # TODO: Change to a further camera view positions
            return self.latest_observation.camera_300

    def enforce_max_episode_length(self, episode_length=2000):
        self.enforce_episode_length = True
        self.max_time_steps = episode_length

    def _is_done(self):
        if self.enforce_episode_length and self.control_index + 1 >= self.max_time_steps:
            return True, 0.0
        else:
            return False, 0.0
