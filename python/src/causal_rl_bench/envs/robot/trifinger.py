from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations
from causal_rl_bench.envs.robot.temp_inverse_kinemetics import apply_inverse_kinemetics
from pybullet_fingers.sim_finger import SimFinger
import numpy as np


class TriFingerRobot(object):
    def __init__(self, action_mode, observation_mode,
                 enable_visualization=True, skip_frame=20,
                 normalize_actions=True, normalize_observations=True,
                 enable_goal_image=False, simulation_time=0.004):
        self.normalize_actions = normalize_actions
        self.normalize_observations = normalize_observations
        self.action_mode = action_mode
        self.enable_goal_image = enable_goal_image
        self.observation_mode = observation_mode
        self.skip_frame = skip_frame
        self.simulation_time = simulation_time
        self.control_index = -1
        self.tri_finger = SimFinger(self.simulation_time, enable_visualization,
                                    "tri")

        self.robot_actions = TriFingerAction(action_mode, normalize_actions)
        self.robot_observations = TriFingerObservations(observation_mode,
                                                        normalize_observations)
        if self.enable_goal_image:
            self.goal_image_instance = SimFinger(self.simulation_time,
                                                 enable_visualization=False,
                                                 finger_type="tri")
            self.goal_image_instance_state = \
                self.goal_image_instance.reset_finger(self.robot_actions.low,
                                                      np.zeros(9, )) #TODO: action modes
        self.last_action = None
        self.last_clipped_action = None
        self.latest_full_state = None
        self.state_size = 18

    def compute_end_effector_positions(self, robot_state):
        tip_positions = self.tri_finger.pinocchio_utils.forward_kinematics(
            robot_state.position
        )
        end_effector_position = np.concatenate(tip_positions)
        return end_effector_position

    def _process_action_joint_positions(self, robot_state):
        last_action_applied = self.get_last_clippd_action()
        if self.normalize_actions and not self.normalize_observations:
            last_action_applied = self.denormalize_observation_for_key(observation=last_action_applied,
                                                                       key='action_joint_positions')
        elif not self.normalize_actions and self.normalize_observations:
            last_action_applied = self.normalize_observation_for_key(observation=last_action_applied,
                                                                     key='action_joint_positions')
        return last_action_applied

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

    def set_skip_frame(self, skip_frame):
        self.skip_frame = skip_frame

    def get_skip_frame(self):
        return self.skip_frame

    def apply_action(self, action):
        self.control_index += 1
        # clip actions to get the one applied
        clipped_action = self.robot_actions.clip_action(action)
        action_to_apply = clipped_action
        if self.normalize_actions:
            action_to_apply = self.robot_actions.denormalize_action(clipped_action)
        if self.action_mode == "joint_positions":
            finger_action = self.tri_finger.Action(position=action_to_apply)
        elif self.action_mode == "joint_torques":
            finger_action = self.tri_finger.Action(torque=action_to_apply)
        elif self.action_mode == "end_effector_positions":
            # joint_positions = self.tri_finger.pybullet_inverse_kinematics([action_to_apply[:3],
            #                                                                action_to_apply[3:6],
            #                                                                action_to_apply[6:]])
            #TODO: only applies it to the first finger to avoid collision for now
            # filtered_action = self.robot_actions.joint_positions_lower_bounds
            # filtered_action[:3] = action_to_apply[:3]
            joint_positions = apply_inverse_kinemetics(self.get_pybullet_client(),
                                                       self.tri_finger.finger_id,
                                                       self.tri_finger.finger_tip_ids,
                                                       action_to_apply,
                                                       list(self.latest_full_state.position))
            finger_action = self.tri_finger.Action(position=joint_positions)
        elif self.action_mode == "delta_end_effector_positions":
            # joint_positions = self.tri_finger.pybullet_inverse_kinematics([action_to_apply[:3],
            #                                                                action_to_apply[3:6],
            #                                                                action_to_apply[6:]])
            #TODO: only applies it to the first finger to avoid collision for now
            # filtered_action = self.robot_actions.joint_positions_lower_bounds
            # filtered_action[:3] = action_to_apply[:3]
            current_end_effector_positions = self.compute_end_effector_positions(self.latest_full_state)
            action_to_apply = current_end_effector_positions + action_to_apply
            joint_positions = apply_inverse_kinemetics(self.get_pybullet_client(),
                                                       self.tri_finger.finger_id,
                                                       self.tri_finger.finger_tip_ids,
                                                       action_to_apply,
                                                       list(self.latest_full_state.position))
            finger_action = self.tri_finger.Action(position=joint_positions)
        else:
            raise Exception("The action mode {} is not supported".
                            format(self.action_mode))
        for _ in range(self.skip_frame):
            t = self.tri_finger.append_desired_action(finger_action)
            self.tri_finger.step_simulation()
        if self.observation_mode == "cameras":
            state = \
                self.tri_finger.get_observation(t, update_images=True)
        else:
            state = \
                self.tri_finger.get_observation(t, update_images=False)
        self.latest_full_state = state
        self.last_action = action
        self.last_clipped_action = clipped_action
        return

    def get_full_state(self):
        return np.append(self.latest_full_state.position,
                         self.latest_full_state.velocity)

    def set_full_state(self, state):
        self.latest_full_state = self.tri_finger.\
            reset_finger(state[:9], state[9:])
        self.last_action = state[:9]
        self.last_clipped_action = state[:9]
        return

    def clear(self):
        self.last_action = None
        self.last_clipped_action = None
        self.latest_full_state = None
        self.control_index = -1
        return

    def reset_state(self, joint_positions=None, joint_velocities=None):
        # This resets the robot fingers into a random position if nothing is provided
        self.latest_full_state = None
        self.control_index = -1
        if joint_positions is None:
            joint_positions = self.sample_positions()
        if joint_velocities is None:
            joint_velocities = np.zeros(9)
        self.latest_full_state = self.tri_finger.reset_finger(joint_positions,
                                                              joint_velocities)
        #TODO: deal with other action types and test them, only dealing with positions here
        self.last_action = joint_positions
        self.last_clipped_action = joint_positions

    def get_last_action(self):
        return self.last_action

    def get_last_clippd_action(self):
        return self.last_clipped_action

    def get_tip_positions(self, robot_state):
        return self.tri_finger.pinocchio_utils.forward_kinematics(
            robot_state.joint_position)
    
    def get_observation_spaces(self):
        return self.robot_observations.get_observation_spaces()

    def sample_positions(self, sampling_strategy="separated"):
        if sampling_strategy == "uniform":
            positions = np.random.uniform(self.robot_actions.joint_positions_lower_bounds,
                                          self.robot_actions.joint_positions_upper_bounds)
        elif sampling_strategy == "separated":
            def sample_point_in_angle_limits():
                while True:
                    joint_pos = np.random.uniform(
                        low=[-np.pi / 2, np.deg2rad(-77.5), np.deg2rad(-172)],
                        high=[np.pi / 2, np.deg2rad(257.5), np.deg2rad(-2)],
                    )
                    tip_pos = self.tri_finger.pinocchio_utils.forward_kinematics(
                        np.concatenate(
                            [joint_pos for i in
                             range(3)]
                        ),
                    )[0]
                    dist_to_center = np.linalg.norm(tip_pos[:2])
                    angle = np.arccos(tip_pos[0] / dist_to_center)
                    if (
                            (np.pi / 6 < angle < 5 / 6 * np.pi)
                            and (tip_pos[1] > 0)
                            and (0.02 < dist_to_center < 0.2)
                            and np.all(self.robot_actions.joint_positions_lower_bounds[0:3] < joint_pos)
                            and np.all(self.robot_actions.joint_positions_upper_bounds[0:3] > joint_pos)
                    ):
                        return joint_pos

            positions = np.concatenate(
                [
                    sample_point_in_angle_limits()
                    for i in range(3)
                ]
            )
        else:
            raise Exception("not yet implemented")
        return positions

    def get_action_spaces(self):
        return self.robot_actions.get_action_space()

    def select_observations(self, observation_keys):
        self.robot_observations.reset_observation_keys()
        for key in observation_keys:
            if key == "end_effector_positions":
                self.robot_observations.add_observation("end_effector_positions",
                                                        observation_fn=self.compute_end_effector_positions)
            elif key == "action_joint_positions":
                self.robot_observations.add_observation("action_joint_positions",
                                                        observation_fn=self._process_action_joint_positions)
            else:
                self.robot_observations.add_observation(key)

    def close(self):
        self.tri_finger.disconnect_from_simulation()
        if self.enable_goal_image:
            self.goal_image_instance.disconnect_from_simulation()

    def get_state_size(self):
        return self.state_size

    #TODO: refactor in the pybullet_fingers repo
    def get_pybullet_client(self):
        return self.tri_finger._p

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None, observation_fn=None):
        self.robot_observations.add_observation(observation_key,
                                                lower_bound,
                                                upper_bound,
                                                observation_fn)

    def get_current_observations(self, helper_keys):
        return self.robot_observations.get_current_observations(self.latest_full_state,
                                                                helper_keys)

    def normalize_observation_for_key(self, observation, key):
        return self.robot_observations.normalize_observation_for_key(observation,
                                                                     key)

    def denormalize_observation_for_key(self, observation, key):
        return self.robot_observations.denormalize_observation_for_key(observation,
                                                                       key)

    def get_current_camera_observations(self):
        return self.robot_observations.get_current_camera_observations(
            self.latest_full_state)

    def get_goal_image_instance_pybullet(self):
        if self.enable_goal_image:
            return self.goal_image_instance
        else:
            raise Exception("goal image is not enabled")

    def get_rest_pose(self):
        deg45 = np.pi / 4

        positions = [0, -deg45, -deg45]
        joint_positions = positions * 3
        end_effector_positions = [0.05142966, 0.03035857, 0.32112874,  0.00057646, -0.05971867,  0.32112874,
                                  -0.05200612,  0.02936011,  0.32112874]
        return joint_positions, end_effector_positions



