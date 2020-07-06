from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations
import numpy as np
import pybullet
from causal_rl_bench.configs.world_constants import WorldConstants


class TriFingerRobot(object):
    def __init__(self, action_mode,
                 observation_mode,
                 skip_frame, normalize_actions,
                 normalize_observations,
                 simulation_time,
                 pybullet_client_full_id,
                 pybullet_client_w_goal_id,
                 pybullet_client_w_o_goal_id,
                 revolute_joint_ids,
                 finger_tip_ids,
                 cameras=None,
                 camera_indicies=np.array([0, 1, 2])):
        """

        :param action_mode:
        :param observation_mode:
        :param skip_frame:
        :param normalize_actions:
        :param normalize_observations:
        :param simulation_time:
        :param pybullet_client_full:
        :param pybullet_client_w_goal:
        :param pybullet_client_w_o_goal:
        """
        self._pybullet_client_full_id = pybullet_client_full_id
        self._pybullet_client_w_goal_id = pybullet_client_w_goal_id
        self._pybullet_client_w_o_goal_id = pybullet_client_w_o_goal_id
        self._revolute_joint_ids = revolute_joint_ids
        self._finger_tip_ids = finger_tip_ids
        self._normalize_actions = normalize_actions
        self._normalize_observations = normalize_observations
        self._action_mode = action_mode
        self._observation_mode = observation_mode
        self._skip_frame = skip_frame
        self._simulation_time = simulation_time
        self._dt = self._simulation_time * self._skip_frame
        #TODO: for some reason this is needed
        self._control_index = -1
        self._position_gains = np.array(
            [10.0, 10.0, 10.0] * 3
        )
        self._velocity_gains = np.array(
            [0.1, 0.3, 0.001] * 3
        )
        self._safety_kd = np.array([0.08, 0.08, 0.04] * 3)
        self._max_motor_torque = 0.36
        self._robot_actions = TriFingerAction(action_mode,
                                              normalize_actions)
        if self._pybullet_client_w_goal_id is not None:
            self._set_finger_state_in_goal_image()
        self._tool_cameras = cameras
        self._camera_indicies = camera_indicies
        self._robot_observations = TriFingerObservations(observation_mode,
                                                         normalize_observations,
                                                         cameras=self._tool_cameras,
                                                         camera_indicies=self._camera_indicies)
        #Take care with the following last action and last clipped action
        # always follow the action mode normalization
        #last_applied_joint_positions is always saved here as
        # denormalized but when its part of the obs space then it follows
        #the observation mode
        self._last_action = None
        self._last_clipped_action = None
        if action_mode != "joint_torques":
            self._last_applied_joint_positions = None
        self._latest_full_state = None
        self._latest_camera_observations = None
        #TODO: repeated
        self._state_size = 18
        #TODO: move this to pybullet_fingers repo maybe?
        #disable velocity control mode
        self._disable_velocity_control()
        return

    def get_link_names(self):
        return WorldConstants.LINK_IDS

    def get_control_index(self):
        return self._control_index

    def get_full_env_state(self):
        return self.get_current_scm_values()

    def set_full_env_state(self, env_state):
        self.apply_interventions(env_state)
        return env_state

    def update_latest_full_state(self):
        if self._pybullet_client_full_id is not None:
            current_joint_states = pybullet.\
                getJointStates(
                WorldConstants.ROBOT_ID, self._revolute_joint_ids,
                 physicsClientId=self._pybullet_client_full_id
            )
        else:
            current_joint_states = pybullet.\
                getJointStates(
                WorldConstants.ROBOT_ID, self._revolute_joint_ids,
                 physicsClientId=self._pybullet_client_w_o_goal_id
            )
        current_position = np.array(
            [joint[0] for joint in current_joint_states]
        )
        current_velocity = np.array(
            [joint[1] for joint in current_joint_states]
        )
        current_torques = np.array(
            [joint[3] for joint in current_joint_states]
        )
        self._latest_full_state = {'positions': current_position,
                                   'velocities': current_velocity,
                                   'torques': current_torques,
                                   'end_effector_positions':
                                       self._compute_end_effector_positions(
                                           current_position)}

        return

    def update_images(self):
        #TODO: this is just a sekelton for now
        observations = []
        observations.append(self._tool_cameras[0].get_image())
        self._latest_camera_observations = observations

    def compute_pd_control_torques(self, joint_positions):
        """
        Compute torque command to reach given target position using a PD
        controller.

        Args:
            joint_positions (array-like, shape=(n,)):  Desired joint positions.

        Returns:
            List of torques to be sent to the joints of the finger in order to
            reach the specified joint_positions.
        """
        position_error = joint_positions - self._latest_full_state['positions']
        position_feedback = np.asarray(self._position_gains) * \
                            position_error
        velocity_feedback = np.asarray(self._velocity_gains) * \
                            self._latest_full_state['velocities']
        joint_torques = position_feedback - velocity_feedback
        self.update_latest_full_state()
        return joint_torques

    def set_action_mode(self, action_mode):
        self._action_mode = action_mode
        self._robot_actions = TriFingerAction(action_mode,
                                              self._normalize_actions)

    def get_upper_joint_positions(self):
        return self._robot_actions.joint_positions_upper_bounds

    def get_action_mode(self):
        return self._action_mode

    def set_observation_mode(self, observation_mode):
        self._observation_mode = observation_mode
        self._robot_observations = \
            TriFingerObservations(observation_mode,
                                  self._normalize_observations,
                                  cameras=self._tool_cameras,
                                  camera_indicies=self._camera_indicies)

    def get_observation_mode(self):
        return self._observation_mode

    def set_skip_frame(self, skip_frame):
        self._skip_frame = skip_frame

    def get_skip_frame(self):
        return self._skip_frame

    def get_full_state(self):
        self.update_latest_full_state()
        return np.append(self._latest_full_state['positions'],
                         self._latest_full_state['velocities'])

    def set_full_state(self, state):
        self._set_finger_state(state[:9], state[9:])
        # here the previous actions will all be zeros to
        # avoid dealing with different action modes for now
        self._last_action = np.zeros(9, )
        self._last_clipped_action = np.zeros(9, )
        if self._action_mode != "joint_torques":
            self._last_applied_joint_positions = list(state[:9])
        return

    def get_last_action(self):
        return self._last_action

    def get_last_clipped_action(self):
        return self._last_clipped_action

    def get_last_applied_joint_positions(self):
        return self._last_applied_joint_positions

    def get_observation_spaces(self):
        return self._robot_observations.get_observation_spaces()

    def get_action_spaces(self):
        return self._robot_actions.get_action_space()

    def get_state_size(self):
        return self._state_size

    def step_simulation(self):
        if self._pybullet_client_full_id is not None:
            pybullet.stepSimulation(
                physicsClientId=self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            pybullet.stepSimulation(
                physicsClientId=self._pybullet_client_w_o_goal_id)
        return

    def apply_action(self, action):
        self._control_index += 1
        clipped_action = self._robot_actions.clip_action(action)
        action_to_apply = clipped_action
        if self._normalize_actions:
            action_to_apply = self._robot_actions.denormalize_action(clipped_action)
        if self._action_mode == "joint_positions":
            self._last_applied_joint_positions = action_to_apply
            for _ in range(self._skip_frame):
                desired_torques = \
                    self.compute_pd_control_torques(action_to_apply)
                self.send_torque_commands(
                    desired_torque_commands=desired_torques)
                self.step_simulation()
        elif self._action_mode == "joint_torques":
            #TODO: deal with clipped action here and observation part too
            for _ in range(self._skip_frame):
                self.send_torque_commands(
                    desired_torque_commands=action_to_apply)
                self.step_simulation()
        elif self._action_mode == "end_effector_positions":
            #TODO: check if the desired tip positions are in the feasible set
            joint_positions = self.get_joint_positions_from_tip_positions\
                (action_to_apply, list(self._latest_full_state['positions']))
            self._last_applied_joint_positions = joint_positions
            for _ in range(self._skip_frame):
                desired_torques = \
                    self.compute_pd_control_torques(joint_positions)
                self.send_torque_commands(
                    desired_torque_commands=desired_torques)
                self.step_simulation()
        else:
            raise Exception("The action mode {} is not supported".
                            format(self._action_mode))
        #now we get the observations
        if self._observation_mode == "cameras":
            self.update_images()
        self._last_action = action
        self._last_clipped_action = clipped_action
        return

    def get_dt(self):
        return self._dt

    def get_latest_full_state(self):
        return self._latest_full_state

    def send_torque_commands(self, desired_torque_commands):
        """

        :param desired_torque_commands:
        :return:
        """
        torque_commands = self._safety_torque_check(desired_torque_commands)
        if self._pybullet_client_w_o_goal_id is not None:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=WorldConstants.ROBOT_ID,
                jointIndices=self._revolute_joint_ids,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=torque_commands,
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
        if self._pybullet_client_full_id is not None:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=WorldConstants.ROBOT_ID,
                jointIndices=self._revolute_joint_ids,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=torque_commands,
                physicsClientId=self._pybullet_client_full_id
            )
        return torque_commands

    def _safety_torque_check(self, desired_torques):
        """

        :param desired_torques:
        :return:
        """
        applied_torques = np.clip(
            np.asarray(desired_torques),
            -self._max_motor_torque,
            +self._max_motor_torque,
        )
        applied_torques -= self._safety_kd * self._latest_full_state['velocities']

        applied_torques = list(
            np.clip(
                np.asarray(applied_torques),
                -self._max_motor_torque,
                +self._max_motor_torque,
            )
        )

        return applied_torques

    def inverse_kinematics(self, desired_tip_positions, rest_pose):
        """

        :param desired_tip_positions:
        :param rest_pose:
        :return:
        """
        desired = np.array(desired_tip_positions)
        desired[2] += WorldConstants.FLOOR_HEIGHT
        desired[5] += WorldConstants.FLOOR_HEIGHT
        desired[8] += WorldConstants.FLOOR_HEIGHT
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        joint_pos = np.zeros([9])
        finger_tip_ids = self._finger_tip_ids
        final_joint_pose = pybullet.calculateInverseKinematics2(WorldConstants.ROBOT_ID,
                                                               [finger_tip_ids[0],
                                                                finger_tip_ids[1],
                                                                finger_tip_ids[2]],
                                                               [desired[0:3],
                                                                desired[3:6],
                                                                desired[6:]],
                                                                solver=pybullet.IK_DLS,
                                                                currentPositions=rest_pose,
                                                                physicsClientId=client)
        joint_pos[:3] = final_joint_pose[:3]
        final_joint_pose = pybullet.calculateInverseKinematics2(WorldConstants.ROBOT_ID,
                                                               [finger_tip_ids[1], finger_tip_ids[0],
                                                                finger_tip_ids[2]],
                                                               [desired[3:6],
                                                                desired[0:3],
                                                                desired[6:]],
                                                                solver=pybullet.IK_DLS,
                                                                currentPositions=rest_pose,
                                                                physicsClientId=client)
        joint_pos[3:6] = final_joint_pose[3:6]
        final_joint_pose = pybullet.calculateInverseKinematics2(WorldConstants.ROBOT_ID,
                                                               [finger_tip_ids[2], finger_tip_ids[0],
                                                                finger_tip_ids[1]],
                                                               [desired[6:],
                                                                desired[0:3],
                                                                desired[3:6]],
                                                                solver=pybullet.IK_DLS,
                                                                currentPositions=rest_pose,
                                                                physicsClientId=client)
        joint_pos[6:] = final_joint_pose[6:]
        if np.isnan(joint_pos).any():
            joint_pos = rest_pose
        return joint_pos

    def get_joint_positions_from_tip_positions(self, tip_positions,
                                               default_pose=None):
        tip_positions[2] += WorldConstants.FLOOR_HEIGHT
        tip_positions[5] += WorldConstants.FLOOR_HEIGHT
        tip_positions[8] += WorldConstants.FLOOR_HEIGHT
        if default_pose is None:
            positions = self.inverse_kinematics(
                tip_positions, list(self.get_rest_pose()[0]))
        else:
            positions = self.inverse_kinematics(
                tip_positions, list(default_pose))
        return positions

    def get_current_camera_observations(self):
        return self._robot_observations.get_current_camera_observations()

    def get_rest_pose(self):
        deg45 = np.pi / 4
        positions = [0, -deg45, -deg45]
        joint_positions = positions * 3
        end_effector_positions = [0.05142966, 0.03035857, 0.32112874,
                                  0.00057646, -0.05971867, 0.32112874,
                                  -0.05200612, 0.02936011, 0.32112874]
        return joint_positions, end_effector_positions

    def get_default_state(self):
        return np.append(self.get_rest_pose()[0],
                         np.zeros(9))

    def get_current_scm_values(self):
        # TODO: not a complete list yet of what we want to expose
        variable_params = dict()
        self.update_latest_full_state()
        variable_params['joint_positions'] = self._latest_full_state['positions']
        variable_params['control_index'] = self._control_index
        variable_params['joint_velocities'] = self._latest_full_state['velocities']
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        position, _ = pybullet. \
            getBasePositionAndOrientation(WorldConstants.ROBOT_ID,
                                          physicsClientId=
                                          client)
        variable_params['robot_height'] = position[-1] + WorldConstants.ROBOT_HEIGHT
        for robot_finger_link in WorldConstants.LINK_IDS:
            variable_params[robot_finger_link] = dict()
            variable_params[robot_finger_link]['color'] = \
                pybullet.getVisualShapeData(WorldConstants.ROBOT_ID,
                                            physicsClientId=client)\
                       [WorldConstants.VISUAL_SHAPE_IDS[robot_finger_link]][7][:3]
            variable_params[robot_finger_link]['mass'] = \
                pybullet.getDynamicsInfo(WorldConstants.ROBOT_ID,
                                         WorldConstants.LINK_IDS[robot_finger_link],
                                         physicsClientId=client)[0]
        return variable_params

    def get_current_observations(self, helper_keys):
        return self._robot_observations.get_current_observations(
            self._latest_full_state, helper_keys)

    def _compute_end_effector_positions(self, joint_positions):
        result = np.array([])
        if self._pybullet_client_full_id is not None:
            position_1 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=5,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_full_id
            )
            position_2 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=10,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_full_id
            )
            position_3 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=15,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_full_id
            )
        else:
            position_1 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=5,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
            position_2 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=10,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
            position_3 = pybullet.getLinkState(
                WorldConstants.ROBOT_ID, linkIndex=15,
                computeForwardKinematics=True,
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
        result = np.append(result, position_1[0])
        result = np.append(result, position_2[0])
        result = np.append(result, position_3[0])
        result[2] -= WorldConstants.FLOOR_HEIGHT
        result[5] -= WorldConstants.FLOOR_HEIGHT
        result[-1] -= WorldConstants.FLOOR_HEIGHT
        # tip_positions = self._pinocchio_utils.forward_kinematics(
        #     joint_positions
        # )
        # result_2 = np.concatenate(tip_positions)
        # result_2[2] -= WorldConstants.FLOOR_HEIGHT
        # result_2[5] -= WorldConstants.FLOOR_HEIGHT
        # result_2[-1] -= WorldConstants.FLOOR_HEIGHT
        return result

    def _process_action_joint_positions(self, robot_state):
        #This returns the absolute joint positions command sent in position control mode
        # (end effector and joint positions)
        #this observation shouldnt be used in torque control
        last_joints_action_applied = self.get_last_applied_joint_positions()
        #always denormalized by default
        if self._normalize_observations:
            last_joints_action_applied = self.normalize_observation_for_key(
                observation=last_joints_action_applied,
                key='action_joint_positions')
        return last_joints_action_applied

    def clear(self):
        self._last_action = np.zeros(9, )
        self._last_clipped_action = np.zeros(9, )
        self.update_latest_full_state()
        self._last_applied_joint_positions = \
            self._latest_full_state['positions']
        self._control_index = -1
        return

    def reset_state(self, joint_positions=None,
                    joint_velocities=None,
                    end_effector_positions=None):
        self._latest_full_state = None
        self._control_index = -1
        if end_effector_positions is not None:
            joint_positions = self.get_joint_positions_from_tip_positions(
                end_effector_positions, list(self.get_rest_pose()[0]))
        if joint_positions is None:
            joint_positions = list(self.get_rest_pose()[0])
        if joint_velocities is None:
            joint_velocities = np.zeros(9)
        self._set_finger_state(joint_positions, joint_velocities)
        #here the previous actions will all be zeros to avoid dealing with different action modes for now
        self._last_action = np.zeros(9, )
        self._last_clipped_action = np.zeros(9, )
        if self._action_mode != "joint_torques":
            self._last_applied_joint_positions = list(joint_positions)
        return

    def sample_joint_positions(self, sampling_strategy="separated"):
        if sampling_strategy == "uniform":
            #TODO: need to check for collisions here and if its feasible or not
            positions = np.random.uniform(self._robot_actions.
                                          joint_positions_lower_bounds,
                                          self._robot_actions.
                                          joint_positions_upper_bounds)
        else:
            raise Exception("not yet implemented")
        return positions

    def sample_end_effector_positions(self, sampling_strategy="from_joints"):
        if sampling_strategy == "middle_stage":
            tip_positions = np.random.uniform(
                [0.1, 0.1, 0.15, 0.1, -0.15, 0.15, -0.15, -0.15, 0.15],
                [0.15, 0.15, 0.15, 0.15, -0.1, 0.15, -0.1, -0.1, 0.15])
            # TODO:add heuristics if the points are in the reachabe sets or not.
            #red is 300, green is 60, blue is 180
        else:
            raise Exception("not yet implemented")
            #perform inverse kinemetics
            #TODO:add heuristics if the points are in the reachabe sets or not.
        return tip_positions

    def forward_simulation(self, time=1):
        n_steps = int(time / self._simulation_time)
        if self._pybullet_client_w_o_goal_id is not None:
            for _ in range(n_steps):
                pybullet.stepSimulation(
                    physicsClientId=self._pybullet_client_w_o_goal_id
                )
        if self._pybullet_client_full_id is not None:
            for _ in range(n_steps):
                pybullet.stepSimulation(
                    physicsClientId=self._pybullet_client_full_id
                )
        return

    def select_observations(self, observation_keys):
        self._robot_observations.reset_observation_keys()
        for key in observation_keys:
            if key == "action_joint_positions":
                self._robot_observations.add_observation(
                    "action_joint_positions",
                    observation_fn=self._process_action_joint_positions)
            else:
                self._robot_observations.add_observation(key)
        return

    def close(self):
        if self._pybullet_client_full_id is not None:
            pybullet.disconnect(
                physicsClientId=self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            pybullet.disconnect(
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
        if self._pybullet_client_w_goal_id is not None:
            pybullet.disconnect(
                physicsClientId=self._pybullet_client_w_goal_id
            )
        return

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None, observation_fn=None):
        self._robot_observations.add_observation(observation_key,
                                                 lower_bound,
                                                 upper_bound,
                                                 observation_fn)

    def normalize_observation_for_key(self, observation, key):
        return self._robot_observations.normalize_observation_for_key(
            observation, key)

    def denormalize_observation_for_key(self, observation, key):
        return self._robot_observations.denormalize_observation_for_key(
            observation, key)

    def apply_interventions(self, interventions_dict):
        #TODO: add friction of each link
        old_state = self.get_full_state()
        if "joint_positions" in interventions_dict:
            new_joint_positions = interventions_dict["joint_positions"]
        else:
            new_joint_positions = old_state[:9]
        if "joint_velocities" in interventions_dict:
            new_joint_velcoities = interventions_dict["joint_velocities"]
        else:
            new_joint_velcoities = old_state[9:]

        if "joint_positions" in interventions_dict or \
                "joint_velocities" in interventions_dict:
            self._set_finger_state(new_joint_positions, new_joint_velcoities)
            self._last_action = np.zeros(9, )
            self._last_clipped_action = np.zeros(9, )
            if self._action_mode != "joint_torques":
                self._last_applied_joint_positions = list(new_joint_positions)
        for intervention in interventions_dict:
            if intervention == "joint_velocities" or \
                    intervention == "joint_positions":
                continue
            if intervention == 'robot_height':
                if self._pybullet_client_w_goal_id is not None:
                    pybullet.resetBasePositionAndOrientation(
                        WorldConstants.ROBOT_ID, [0, 0, interventions_dict[intervention] - WorldConstants.ROBOT_HEIGHT],
                        [0, 0, 0, 1],
                        physicsClientId=
                        self._pybullet_client_w_goal_id)
                if self._pybullet_client_w_o_goal_id is not None:
                    pybullet.resetBasePositionAndOrientation(
                        WorldConstants.ROBOT_ID,  [0, 0, interventions_dict[intervention] - WorldConstants.ROBOT_HEIGHT],
                        [0, 0, 0, 1],
                        physicsClientId=
                        self._pybullet_client_w_o_goal_id)
                if self._pybullet_client_full_id is not None:
                    pybullet.resetBasePositionAndOrientation(
                        WorldConstants.ROBOT_ID,  [0, 0, interventions_dict[intervention] - WorldConstants.ROBOT_HEIGHT],
                        [0, 0, 0, 1],
                        physicsClientId=
                        self._pybullet_client_full_id)
                continue
            if "robot_finger" in intervention:
                for sub_intervention_variable in \
                        interventions_dict[intervention]:
                    if sub_intervention_variable == 'color':
                        if self._pybullet_client_w_goal_id is not None:
                            pybullet.changeVisualShape(
                                WorldConstants.ROBOT_ID,
                                WorldConstants.LINK_IDS[intervention],
                                rgbaColor=np.append(
                                    interventions_dict[intervention]
                                    [sub_intervention_variable], 1),
                                physicsClientId=self._pybullet_client_w_goal_id)
                        if self._pybullet_client_w_o_goal_id is not None:
                            pybullet.changeVisualShape(
                                WorldConstants.ROBOT_ID,
                                WorldConstants.LINK_IDS[intervention],
                                rgbaColor=np.append(
                                    interventions_dict[intervention]
                                    [sub_intervention_variable], 1),
                                physicsClientId=self._pybullet_client_w_o_goal_id)
                        if self._pybullet_client_full_id is not None:
                            pybullet.changeVisualShape(
                                WorldConstants.ROBOT_ID,
                                WorldConstants.LINK_IDS[intervention],
                                rgbaColor=np.append(
                                    interventions_dict[intervention]
                                    [sub_intervention_variable], 1),
                                physicsClientId=self._pybullet_client_full_id)
                    elif sub_intervention_variable == 'mass':
                        if self._pybullet_client_w_o_goal_id is not None:
                            pybullet.changeDynamics \
                                (WorldConstants.ROBOT_ID,
                                 WorldConstants.LINK_IDS[intervention], mass=
                                 interventions_dict[intervention]
                                 [sub_intervention_variable],
                                 physicsClientId=self._pybullet_client_w_o_goal_id)
                        if self._pybullet_client_full_id is not None:
                            pybullet.changeDynamics \
                                (WorldConstants.ROBOT_ID,
                                 WorldConstants.LINK_IDS[intervention], mass=
                                 interventions_dict[intervention]
                                 [sub_intervention_variable],
                                 physicsClientId=self._pybullet_client_full_id)
                    else:
                        raise Exception(
                            "The intervention state variable specified is "
                            "not allowed")
            elif intervention == "control_index":
                self._control_index = interventions_dict["control_index"]
            else:
                raise Exception("The intervention state variable specified is "
                                "not allowed", intervention)
        self.update_latest_full_state()
        return

    def check_feasibility_of_robot_state(self):
        """
        This function checks the feasibility of the current state of the robot
        (i.e checks if its in penetration with anything now
        Parameters
        ---------

        Returns
        -------
            feasibility_flag: bool
                A boolean indicating whether the stage is in a collision state
                or not.
        """
        if self._pybullet_client_full_id is not None:
            client = self._pybullet_client_full_id
        else:
            client = self._pybullet_client_w_o_goal_id
        for contact in pybullet.getContactPoints(physicsClientId=client):
            if (contact[1] == WorldConstants.ROBOT_ID or
                contact[2] == WorldConstants.ROBOT_ID) and \
                    contact[8] < -0.0095:
                return False
        return True

    def is_self_colliding(self):
        if self._pybullet_client_full_id is not None:
            client = self._pybullet_client_full_id
        else:
            client = self._pybullet_client_w_o_goal_id
        for contact in pybullet.getContactPoints(physicsClientId=client):
            if contact[1] == WorldConstants.ROBOT_ID and \
                    contact[2] == WorldConstants.ROBOT_ID:
                return True
        return False

    def is_colliding_with_stage(self):
        if self._pybullet_client_full_id is not None:
            client = self._pybullet_client_full_id
        else:
            client = self._pybullet_client_w_o_goal_id
        for contact in pybullet.getContactPoints(physicsClientId=client):
            if (contact[1] == WorldConstants.ROBOT_ID and contact[2]
                == WorldConstants.STAGE_ID) or \
                    (contact[2] == WorldConstants.ROBOT_ID and contact[1]
                     == WorldConstants.STAGE_ID):
                return True
        return False

    def is_in_contact_with_block(self, block):
        if self._pybullet_client_full_id is not None:
            client = self._pybullet_client_full_id
        else:
            client = self._pybullet_client_w_o_goal_id
        for contact in pybullet.getContactPoints(physicsClientId=client):
            if (contact[1] == WorldConstants.ROBOT_ID and
                contact[2] == block._block_ids[0]) or \
                    (contact[2] == WorldConstants.ROBOT_ID and
                     contact[1] == block._block_ids[0]):
                return True
        return False

    def get_normal_interaction_force_with_block(self, block,
                                                finger_tip_number):
        # TODO: doesnt account for several contacts per body
        if self._pybullet_client_full_id is not None:
            client = self._pybullet_client_full_id
        else:
            client = self._pybullet_client_w_o_goal_id
        if finger_tip_number == 60:
            idx = WorldConstants.LINK_IDS['robot_finger_60_link_3']
        elif finger_tip_number == 120:
            idx = WorldConstants.LINK_IDS['robot_finger_120_link_3']
        elif finger_tip_number == 300:
            idx = WorldConstants.LINK_IDS['robot_finger_300_link_3']
        else:
            raise Exception("finger tip number doesnt exist")

        for contact in pybullet.getContactPoints(physicsClientId=client):
            if (contact[1] == WorldConstants.ROBOT_ID and
                contact[2] == block._block_ids[0]) or \
                    (contact[2] == WorldConstants.ROBOT_ID
                     and contact[1] == block._block_ids[0]):
                return contact[9]*np.array(contact[7])
            for contact in pybullet.getContactPoints(physicsClientId=client):
                if (contact[1] == WorldConstants.ROBOT_ID
                    and contact[2] == block._block_ids[0]
                    and contact[3] == idx) or  \
                        (contact[2] == WorldConstants.ROBOT_ID
                         and contact[1] == block._block_ids[0]
                     and contact[4] == idx):
                    return contact[9] * np.array(contact[7])
        return None

    def get_tip_contact_states(self):
        #TODO: only support open and closed states (should support slipping too)
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id is not None
        else:
            client = self._pybullet_client_full_id
        contact_tips = [0, 0, 0] #all are open
        for contact in pybullet.getContactPoints(physicsClientId=client):
            if contact[1] == WorldConstants.ROBOT_ID:
                if contact[3] == WorldConstants.LINK_IDS['robot_finger_60_link_3']:
                    contact_tips[0] = 1
                elif contact[3] == WorldConstants.LINK_IDS['robot_finger_120_link_3']:
                    contact_tips[1] = 1
                elif contact[3] == WorldConstants.LINK_IDS['robot_finger_300_link_3']:
                    contact_tips[2] = 1
            elif contact[2] == WorldConstants.ROBOT_ID:
                if contact[4] == WorldConstants.LINK_IDS['robot_finger_60_link_3']:
                    contact_tips[0] = 1
                elif contact[4] == WorldConstants.LINK_IDS['robot_finger_180_link_3']:
                    contact_tips[1] = 1
                elif contact[4] == WorldConstants.LINK_IDS['robot_finger_300_link_3']:
                    contact_tips[2] = 1
        return contact_tips

    def _disable_velocity_control(self):
        """
        To disable the high friction velocity motors created by
        default at all revolute and prismatic joints while loading them from
        the urdf.
        """
        if self._pybullet_client_full_id is not None:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=WorldConstants.ROBOT_ID,
                jointIndices=self._revolute_joint_ids,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocities=[0] * len(self._revolute_joint_ids),
                forces=[0] * len(self._revolute_joint_ids),
                physicsClientId=self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=WorldConstants.ROBOT_ID,
                jointIndices=self._revolute_joint_ids,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocities=[0] * len(self._revolute_joint_ids),
                forces=[0] * len(self._revolute_joint_ids),
                physicsClientId=
                self._pybullet_client_w_o_goal_id
            )

    def _set_finger_state(self, joint_positions,
                         joint_velocities=None):
        if self._pybullet_client_full_id is not None:
            if joint_velocities is None:
                for i, joint_id in enumerate(self._revolute_joint_ids):
                    pybullet.resetJointState(
                        WorldConstants.ROBOT_ID, joint_id, joint_positions[i],
                        physicsClientId =self._pybullet_client_full_id
                    )
            else:
                for i, joint_id in enumerate(self._revolute_joint_ids):
                    pybullet.resetJointState(
                        WorldConstants.ROBOT_ID, joint_id, joint_positions[i],
                        joint_velocities[i],
                        physicsClientId=self._pybullet_client_full_id
                    )

        if self._pybullet_client_w_o_goal_id is not None:
            if joint_velocities is None:
                for i, joint_id in enumerate(self._revolute_joint_ids):
                    pybullet.resetJointState(
                        WorldConstants.ROBOT_ID, joint_id, joint_positions[i],
                        physicsClientId=self._pybullet_client_w_o_goal_id
                    )
            else:
                for i, joint_id in enumerate(self._revolute_joint_ids):
                    pybullet.resetJointState(
                        WorldConstants.ROBOT_ID, joint_id, joint_positions[i],
                        joint_velocities[i],
                        physicsClientId=self._pybullet_client_w_o_goal_id
                    )
        self.update_latest_full_state()
        return

    def _set_finger_state_in_goal_image(self):
        joint_positions = \
            self._robot_actions.joint_positions_lower_bounds
        for i, joint_id in enumerate(self._revolute_joint_ids):
            pybullet.resetJointState(
                WorldConstants.ROBOT_ID, joint_id, joint_positions[i],
                physicsClientId=self._pybullet_client_w_goal_id
            )
        return
