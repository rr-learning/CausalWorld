from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations

from pybullet_fingers.sim_finger import SimFinger
import numpy as np


class TriFingerRobot(object):
    def __init__(self, action_mode, observation_mode, enable_visualization=True,
                 camera_skip_frame=40, skip_frame=20, camera_turned_on=False,
                 normalize_actions=True, normalize_observations=True):
        self.normalize_actions = normalize_actions
        self.normalize_observations = normalize_observations
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.camera_skip_frame = camera_skip_frame
        self.skip_frame = skip_frame
        self.camera_turned_on = camera_turned_on
        self.simulation_time = 0.001
        self.control_index = -1
        if self.camera_turned_on:
            assert ((float(self.camera_skip_frame) / self.skip_frame).is_integer())
        self.tri_finger = SimFinger(self.simulation_time, enable_visualization,
                                    "tri")
        self.robot_actions = TriFingerAction(action_mode, normalize_actions)
        self.robot_observations = TriFingerObservations(observation_mode)


        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None
        self.state_size = 18

    def add_end_effector_postions_observations(self):
        self.robot_observations.add_observation("end_effector_positions",
                                                observation_fn=self._compute_end_effector_positions)

    def _compute_end_effector_positions(self, robot_state):
        tip_positions = self.tri_finger.pinocchio_utils.forward_kinematics(
            robot_state.position
        )
        end_effector_position = np.concatenate(tip_positions)
        return end_effector_position

    def set_action_mode(self, action_mode):
        self.action_mode = action_mode
        self.robot_actions = TriFingerAction(action_mode,
                                             self.normalize_actions)

    def get_action_mode(self):
        return self.action_mode

    def set_observation_mode(self, observation_mode):
        self.observation_mode = observation_mode
        self.robot_observations = \
            TriFingerObservations(observation_mode, self.normalize_observations)

    def get_observation_mode(self):
        return self.observation_mode

    def set_camera_skip_frame(self, camera_skip_frame):
        self.camera_skip_frame = camera_skip_frame
        if self.camera_turned_on:
            assert ((float(self.camera_skip_frame) / self.skip_frame).is_integer())

    def get_camera_skip_frame(self):
        return self.camera_skip_frame

    def set_skip_frame(self, skip_frame):
        self.skip_frame = skip_frame
        if self.camera_turned_on:
            assert ((float(self.camera_skip_frame)/self.skip_frame).is_integer())

    def get_skip_frame(self):
        return self.skip_frame

    def turn_on_cameras(self):
        self.camera_turned_on = True
        self.robot_observations.add_observation("cameras")
        assert ((float(self.camera_skip_frame) / self.skip_frame).is_integer())

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

        for _ in range(self.skip_frame):
            t = self.tri_finger.append_desired_action(finger_action)
            self.tri_finger.step_simulation()
        if self.camera_turned_on and \
                self.control_index % self.camera_skip_frame == 0:
            state = \
                self.tri_finger.get_observation(t, update_images=True)
        else:
            state = \
                self.tri_finger.get_observation(t, update_images=False)
        observations_dict = self.robot_observations.get_current_observations(
            state)
        self.latest_full_state = state
        self.last_action_applied = action
        self.latest_observation = observations_dict
        return observations_dict

    def get_full_state(self):
        # The full state is independent of the observation mode
        #TODO: only returning positions since we cant set the velocity for now
        # full_state = {"joint_positions": self.latest_full_state.position,
        #               "joint_velocities": self.latest_full_state.velocity,
        #               "joint_torques": self.latest_full_state.torque}

        return np.append(self.latest_full_state.position,
                         self.latest_full_state.velocity)

    def set_full_state(self, state):
        self.latest_full_state = self.tri_finger.\
            reset_finger(state[:9], state[9:])
        observations_dict = self.robot_observations.get_current_observations(
            self.latest_full_state)
        self.latest_observation = observations_dict
        return

    def clear(self):
        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None
        self.control_index = -1
        return

    def reset_state(self, joint_positions=None, joint_velocities=None):
        # This resets the robot fingers into a random position if nothing is provided
        self.last_action_applied = None
        self.latest_observation = None
        self.latest_full_state = None
        self.control_index = -1
        if joint_positions is None:
            joint_positions = self.robot_actions.sample_actions()
        if joint_velocities is None:
            joint_velocities = np.zeros(9)
        self.latest_full_state = self.tri_finger.reset_finger(joint_positions,
                                                              joint_velocities)
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
        current_observations_keys = list(self.robot_observations.
                                         observations_keys)
        for key in current_observations_keys:
            if key not in observation_keys:
                self.robot_observations.remove_observations([key])

    def close(self):
        self.tri_finger.disconnect_from_simulation()

    def get_state_size(self):
        return self.state_size

    #TODO: refactor in the pybullet_fingers repo
    def get_pybullet_client(self):
        return self.tri_finger._p


