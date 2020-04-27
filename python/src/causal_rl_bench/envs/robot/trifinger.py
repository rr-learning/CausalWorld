from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations

from pybullet_fingers.sim_finger import SimFinger
import numpy as np


class TriFingerRobot(object):
    def __init__(self, action_mode, observation_mode, enable_visualization=True,
                 camera_rate=0.3, control_rate=0.02, camera_turned_on=False,
                 normalize_actions=True):
        self.normalized_actions = normalize_actions
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.camera_rate = camera_rate
        self.control_rate = control_rate
        self.camera_turned_on = camera_turned_on
        self.simulation_rate = 0.001
        self.control_index = -1

        self.tri_finger = SimFinger(self.simulation_rate, enable_visualization, "tri")

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

    def set_action_mode(self, action_mode):
        self.robot_actions = TriFingerAction(action_mode)

    def get_action_mode(self):
        return self.action_mode

    def set_observation_mode(self, observation_mode):
        self.robot_observations = TriFingerObservations(observation_mode)

    def get_observation_mode(self):
        return self.observation_mode

    def set_camera_rate(self, camera_rate):
        self.camera_rate = camera_rate

    def get_camera_rate(self):
        return self.camera_rate

    def set_control_rate(self, control_rate):
        self.control_rate = control_rate

    def get_control_rate(self):
        return self.control_rate

    def turn_on_cameras(self):
        self.camera_turned_on = True

    def turn_off_cameras(self):
        self.camera_turned_on = False

    def apply_action(self, action):
        self.control_index += 1
        unscaled_action = self.robot_actions.denormalize_action(action)
        if self.action_mode == "joint_positions":
            finger_action = self.tri_finger.Action(position=unscaled_action)
        elif self.action_mode == "torques":
            finger_action = self.tri_finger.Action(torque=unscaled_action)
        elif self.action_mode == "both":
            finger_action = self.tri_finger.Action(torque=unscaled_action[:9], position=unscaled_action[9:])
        else:
            finger_action = self.tri_finger.Action(position=unscaled_action)

        for _ in range(self.steps_per_control):
            t = self.tri_finger.append_desired_action(finger_action)
            self.tri_finger.step_simulation()
        if self.camera_turned_on and self.control_index % self.camera_skip_steps == 0:
            observation = self.tri_finger.get_observation(t, update_images=True)
        else:
            observation = self.tri_finger.get_observation(t, update_images=False)
        self.latest_observation = observation
        self.last_action_applied = action

    def get_full_state(self):
        # The full state is independent of the observation mode
        full_state = {"tf_positions": self.latest_observation.position,
                      "tf_velocities": self.latest_observation.velocity,
                      "tf_torque": self.latest_observation.torque}
        return full_state

    def set_full_state(self, joint_positions):
        self.tri_finger.reset_finger(joint_positions)

    def reset_robot_state(self):
        # This resets the robot fingers into the base state
        self.last_action_applied = None
        self.control_index = -1
        joint_positions = self.robot_observations.lower_bounds["joint_positions"]
        self.latest_observation = self.tri_finger.reset_finger(joint_positions)

    def get_last_action_applied(self):
        return self.last_action_applied

    def get_current_full_observations(self):
        if self.observation_mode == "structured":
            return np.concatenate(self.latest_observation.position,
                                  self.latest_observation.velocity,
                                  self.latest_observation.torque)
        elif self.observation_mode == "cameras":
            latest_camera_observations = [self.latest_observation.camera_60,
                                          self.latest_observation.camera_180,
                                          self.latest_observation.camera_300]
            return np.stack(latest_camera_observations, axis=0)

    def get_current_partial_observations(self, keys):
        raise Exception("Not implemented")
