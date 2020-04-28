from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations

from pybullet_fingers.sim_finger import SimFinger
import numpy as np


class TriFingerRobot(object):
    def __init__(self, action_mode, observation_mode, enable_visualization=True,
                 camera_rate=0.3, control_rate=0.02, camera_turned_on=False,
                 normalize_actions=True, normalize_observations=True):
        self.normalize_actions = normalize_actions
        self.normalize_observations = normalize_observations
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.camera_rate = camera_rate
        self.control_rate = control_rate
        self.camera_turned_on = camera_turned_on
        self.simulation_rate = 0.001
        self.control_index = -1

        self.tri_finger = SimFinger(self.simulation_rate, enable_visualization,
                                    "tri")

        self.camera_skip_steps = int(round(self.camera_rate / self.control_rate))
        self.steps_per_control = int(round(self.control_rate / self.simulation_rate))
        assert (abs(self.control_rate - self.steps_per_control * self.simulation_rate)
                <= 0.000001)
        assert (abs(self.camera_rate - self.camera_skip_steps * self.control_rate)
                <= 0.000001)

        self.robot_actions = TriFingerAction(action_mode, normalize_actions)
        self.robot_observations = TriFingerObservations(observation_mode)

        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None

    def set_action_mode(self, action_mode):
        self.robot_actions = TriFingerAction(action_mode,
                                             self.normalize_actions)

    def get_action_mode(self):
        return self.action_mode

    def set_observation_mode(self, observation_mode):
        self.robot_observations = \
            TriFingerObservations(observation_mode, self.normalize_observations)

    def get_observation_mode(self):
        return self.observation_mode

    def set_camera_rate(self, camera_rate):
        self.camera_rate = camera_rate
        self.camera_skip_steps = int(
            round(self.camera_rate / self.control_rate))
        assert (abs(
            self.camera_rate - self.camera_skip_steps * self.control_rate)
                <= 0.000001)

    def get_camera_rate(self):
        return self.camera_rate

    def set_control_rate(self, control_rate):
        self.control_rate = control_rate
        self.steps_per_control = int(
            round(self.control_rate / self.simulation_rate))
        assert (abs(
            self.control_rate - self.steps_per_control * self.simulation_rate)
                <= 0.000001)

    def get_control_rate(self):
        return self.control_rate

    def turn_on_cameras(self):
        self.camera_turned_on = True
        self.robot_observations.add_observation("cameras")

    def turn_off_cameras(self):
        self.camera_turned_on = False
        self.robot_observations.remove_observations(["cameras"])

    def apply_action(self, action):
        self.control_index += 1
        unscaled_action = self.robot_actions.denormalize_action(action)
        if self.action_mode == "joint_positions":
            finger_action = self.tri_finger.Action(position=unscaled_action)
        elif self.action_mode == "joint_torques":
            finger_action = self.tri_finger.Action(torque=unscaled_action)
        else:
            raise Exception("The action mode {} is not supported".
                            format(self.action_mode))

        for _ in range(self.steps_per_control):
            t = self.tri_finger.append_desired_action(finger_action)
            self.tri_finger.step_simulation()
        if self.camera_turned_on and \
                self.control_index % self.camera_skip_steps == 0:
            state = \
                self.tri_finger.get_observation(t, update_images=True)
        else:
            state = \
                self.tri_finger.get_observation(t, update_images=False)
        observations_dict, observations_list = \
            self.robot_observations.get_current_observations(state)
        self.latest_full_state = state
        self.last_action_applied = action
        self.latest_observation = observations_dict
        return observations_dict

    def get_full_state(self):
        # The full state is independent of the observation mode
        full_state = {"joint_positions": self.latest_full_state.position,
                      "joint_velocities": self.latest_full_state.velocity,
                      "joint_torques": self.latest_full_state.torque}

        return full_state

    def set_full_state(self, joint_positions):
        self.latest_full_state = self.tri_finger.reset_finger(joint_positions)
        observations_dict, observations_list = \
            self.robot_observations.get_current_observations \
                (self.latest_full_state)
        self.latest_observation = observations_dict
        return

    def clear(self):
        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None
        self.control_index = -1
        return

    def reset_state(self):
        # This resets the robot fingers into the base state
        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None
        self.control_index = -1
        # TODO: reset to random position? when specified?
        joint_positions = self.robot_observations.lower_bounds["joint_positions"]
        self.latest_full_state = self.tri_finger.reset_finger(joint_positions)
        observations_dict, observations_list = \
            self.robot_observations.get_current_observations\
                (self.latest_full_state)
        self.latest_observation = observations_dict

    def get_last_action_applied(self):
        return self.last_action_applied

    def get_current_full_observations(self):
        return self.latest_observation

    def get_current_partial_observations(self, keys):
        raise Exception("Not implemented")

    def get_tip_positions(self, robot_state):
        return self.tri_finger.pinocchio_utils.forward_kinematics(
            robot_state.joint_position)
    
    def get_observation_spaces(self):
        return self.robot_observations.get_observation_spaces()

    def sample_actions(self, sampling_strategy="uniform"):
        self.robot_actions.sample_actions(sampling_strategy)

    def sample_positions(self, sampling_strategy="uniform"):
        positions = self.robot_actions.sample_actions(sampling_strategy,
                                                      mode="joint_positions")
        return positions

    def get_action_spaces(self):
        return self.robot_actions.get_action_space()

    def select_observations(self, observation_keys):
        current_observations_keys = self.robot_observations.observations_keys
        for key in current_observations_keys:
            if key not in observation_keys:
                self.robot_observations.remove_observations([key])
