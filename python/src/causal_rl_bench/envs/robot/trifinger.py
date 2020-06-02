from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations
from pybullet_fingers.sim_finger import SimFinger
import numpy as np
import math


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
        self.dt = self.simulation_time * self.skip_frame
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
                self.goal_image_instance.reset_finger(
                    self.robot_actions.joint_positions_lower_bounds,
                    np.zeros(9, ))
        #Take care with the following last action and last clipped action always follow the action mode normalization
        #last_applied_joint_positions is always saved here as denormalized but when its part of the obs space then it follows
        #the observation mode
        self.last_action = None
        self.last_clipped_action = None
        if action_mode != "joint_torques":
            self.last_applied_joint_positions = None
        self.latest_full_state = None
        self.state_size = 18
        #TODO: move this to pybullet_fingers repo maybe?
        self.link_ids = {'robot_finger_60_link_0': 1,
                         'robot_finger_60_link_1': 2,
                         'robot_finger_60_link_2': 3,
                         'robot_finger_60_link_3': 4,
                         'robot_finger_120_link_0': 6,
                         'robot_finger_120_link_1': 7,
                         'robot_finger_120_link_2': 8,
                         'robot_finger_120_link_3': 9,
                         'robot_finger_300_link_0': 11,
                         'robot_finger_300_link_1': 12,
                         'robot_finger_300_link_2': 13,
                         'robot_finger_300_link_3': 14}
        self.visual_shape_ids_ids = {'robot_finger_60_link_0': 0,
                                     'robot_finger_60_link_1': 1,
                                     'robot_finger_60_link_2': 2,
                                     'robot_finger_60_link_3': 3,
                                     'robot_finger_120_link_0': 4,
                                     'robot_finger_120_link_1': 5,
                                     'robot_finger_120_link_2': 6,
                                     'robot_finger_120_link_3': 7,
                                     'robot_finger_300_link_0': 8,
                                     'robot_finger_300_link_1': 9,
                                     'robot_finger_300_link_2': 10,
                                     'robot_finger_300_link_3': 11}
        return

    def set_action_mode(self, action_mode):
        self.action_mode = action_mode
        self.robot_actions = TriFingerAction(action_mode,
                                             self.normalize_actions)

    def get_action_mode(self):
        return self.action_mode

    def set_observation_mode(self, observation_mode):
        self.observation_mode = observation_mode
        self.robot_observations = \
            TriFingerObservations(observation_mode,
                                  self.normalize_observations)

    def get_observation_mode(self):
        return self.observation_mode

    def set_skip_frame(self, skip_frame):
        self.skip_frame = skip_frame

    def get_skip_frame(self):
        return self.skip_frame

    def get_full_state(self):
        return np.append(self.latest_full_state.position,
                         self.latest_full_state.velocity)

    def set_full_state(self, state):
        self.latest_full_state = self.tri_finger.\
            reset_finger(state[:9], state[9:])
        # here the previous actions will all be zeros to avoid dealing with different action modes for now
        self.last_action = np.zeros(9, )
        self.last_clipped_action = np.zeros(9, )
        if self.action_mode != "joint_torques":
            self.last_applied_joint_positions = list(state[:9])
        return

    def get_last_action(self):
        return self.last_action

    def get_last_clipped_action(self):
        return self.last_clipped_action

    def get_last_applied_joint_positions(self):
        return self.last_applied_joint_positions

    def get_observation_spaces(self):
        return self.robot_observations.get_observation_spaces()

    def get_action_spaces(self):
        return self.robot_actions.get_action_space()

    def get_state_size(self):
        return self.state_size

    #TODO: refactor in the pybullet_fingers repo
    def get_pybullet_client(self):
        return self.tri_finger._p

    def apply_action(self, action):
        self.control_index += 1
        clipped_action = self.robot_actions.clip_action(action)
        action_to_apply = clipped_action
        if self.normalize_actions:
            action_to_apply = self.robot_actions.denormalize_action(clipped_action)
        if self.action_mode == "joint_positions":
            finger_action = self.tri_finger.Action(position=action_to_apply)
            self.last_applied_joint_positions = action_to_apply
        elif self.action_mode == "joint_torques":
            #TODO: deal with clipped action here and observation part too
            finger_action = self.tri_finger.Action(torque=action_to_apply)
        elif self.action_mode == "end_effector_positions":
            #TODO: check if the desired tip positions are in the feasible set
            joint_positions = self.get_joint_positions_from_tip_positions\
                (action_to_apply, list(self.latest_full_state.position))
            finger_action = self.tri_finger.Action(position=joint_positions)
            self.last_applied_joint_positions = joint_positions
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

    def get_joint_positions_from_tip_positions(self, tip_positions,
                                               default_pose=None):
        if default_pose is None:
            positions = self.tri_finger.pybullet_inverse_kinematics(
                tip_positions, list(self.get_rest_pose()[0]))
        else:
            positions = self.tri_finger.pybullet_inverse_kinematics(
                tip_positions, default_pose)
        return positions

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
        end_effector_positions = [0.05142966, 0.03035857, 0.32112874,
                                  0.00057646, -0.05971867, 0.32112874,
                                  -0.05200612, 0.02936011, 0.32112874]
        return joint_positions, end_effector_positions

    def get_default_state(self):
        return np.append(self.get_rest_pose()[0],
                         np.zeros(9))

    def get_current_variables_values(self):
        # TODO: not a complete list yet of what we want to expose
        variable_params = dict()
        if self.is_initialized():
            state = self.get_full_state()
        else:
            state = self.get_default_state()
        variable_params['joint_positions'] = state[:9]
        variable_params['joint_velocities'] = state[9:]
        for robot_finger_link in self.link_ids:
            variable_params[robot_finger_link] = dict()
            variable_params[robot_finger_link]['color'] = \
                self.get_pybullet_client(). \
                    getVisualShapeData(self.tri_finger.finger_id) \
                    [self.visual_shape_ids_ids[robot_finger_link]][7][:3]
            variable_params[robot_finger_link]['mass'] = \
                self.get_pybullet_client(). \
                    getDynamicsInfo(self.tri_finger.finger_id,
                                    self.link_ids[robot_finger_link])[0]
        return variable_params

    def get_current_observations(self, helper_keys):
        return self.robot_observations.get_current_observations(
            self.latest_full_state, helper_keys)

    def compute_end_effector_positions(self, joint_positions):
        tip_positions = self.tri_finger.pinocchio_utils.forward_kinematics(
            joint_positions
        )
        end_effector_position = np.concatenate(tip_positions)
        return end_effector_position

    def _compute_end_effector_positions(self, robot_state):
        tip_positions = self.tri_finger.pinocchio_utils.forward_kinematics(
            robot_state.position
        )
        end_effector_position = np.concatenate(tip_positions)
        return end_effector_position

    def _process_action_joint_positions(self, robot_state):
        #This returns the absolute joint positions command sent in position control mode
        # (end effector and joint positions)
        #this observation shouldnt be used in torque control
        last_joints_action_applied = self.get_last_applied_joint_positions()
        #always denormalized by default
        if self.normalize_observations:
            last_joints_action_applied = self.normalize_observation_for_key(
                observation=last_joints_action_applied,
                key='action_joint_positions')
        return last_joints_action_applied

    def clear(self):
        self.last_action = None
        self.last_clipped_action = None
        self.last_applied_joint_positions = None
        self.latest_full_state = None
        self.control_index = -1
        return

    def reset_state(self, joint_positions=None,
                    joint_velocities=None,
                    end_effector_positions=None):
        self.latest_full_state = None
        self.control_index = -1
        if end_effector_positions is not None:
            joint_positions = self.get_joint_positions_from_tip_positions(
                end_effector_positions, list(self.get_rest_pose()[0]))
        if joint_positions is None:
            joint_positions = list(self.get_rest_pose()[0])
        if joint_velocities is None:
            joint_velocities = np.zeros(9)
        self.latest_full_state = self.tri_finger.reset_finger(joint_positions,
                                                              joint_velocities)
        #here the previous actions will all be zeros to avoid dealing with different action modes for now
        self.last_action = np.zeros(9,)
        self.last_clipped_action = np.zeros(9,)
        if self.action_mode != "joint_torques":
            self.last_applied_joint_positions = list(joint_positions)
        return

    def sample_joint_positions(self, sampling_strategy="separated"):
        if sampling_strategy == "uniform":
            #TODO: need to check for collisions here and if its feasible or not
            positions = np.random.uniform(self.robot_actions.
                                          joint_positions_lower_bounds,
                                          self.robot_actions.
                                          joint_positions_upper_bounds)
        elif sampling_strategy == "separated":
            #TODO: double check this function
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
                            and np.all(self.robot_actions.
                                               joint_positions_lower_bounds
                                       [0:3] < joint_pos)
                            and np.all(self.robot_actions.
                                               joint_positions_upper_bounds
                                       [0:3] > joint_pos)
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

    def sample_end_effector_positions(self, sampling_strategy="from_joints"):
        if sampling_strategy == "middle_stage":
            tip_positions = np.random.uniform(
                [0.1, 0.1, 0.15, 0.1, -0.15, 0.15, -0.15, -0.15, 0.15],
                [0.15, 0.15, 0.15, 0.15, -0.1, 0.15, -0.1, -0.1, 0.15])
            # TODO:add heuristics if the points are in the reachabe sets or not.
            #red is 300, green is 60, blue is 180
        elif sampling_strategy == "from_joints":
            joints_goal = self.sample_joint_positions()
            tip_positions = self.\
                compute_end_effector_positions(joints_goal)
        else:
            raise Exception("not yet implemented")
            #perform inverse kinemetics
            #TODO:add heuristics if the points are in the reachabe sets or not.
        return tip_positions

    def forward_simulation(self, time=1):
        n_steps = int(time / self.simulation_time)
        for _ in range(n_steps):
            self.get_pybullet_client().stepSimulation()
        return

    def select_observations(self, observation_keys):
        self.robot_observations.reset_observation_keys()
        for key in observation_keys:
            if key == "end_effector_positions":
                self.robot_observations.add_observation(
                    "end_effector_positions",
                    observation_fn=self._compute_end_effector_positions)
            elif key == "action_joint_positions":
                #check that its possible
                if self.action_mode == "joint_torques":
                    raise Exception("action_joint_positions is not supported "
                                    "for joint torques action mode")
                self.robot_observations.add_observation(
                    "action_joint_positions",
                    observation_fn=self._process_action_joint_positions)
            else:
                self.robot_observations.add_observation(key)
        return

    def close(self):
        self.tri_finger.disconnect_from_simulation()
        if self.enable_goal_image:
            self.goal_image_instance.disconnect_from_simulation()

    def add_observation(self, observation_key, lower_bound=None,
                        upper_bound=None, observation_fn=None):
        self.robot_observations.add_observation(observation_key,
                                                lower_bound,
                                                upper_bound,
                                                observation_fn)

    def normalize_observation_for_key(self, observation, key):
        return self.robot_observations.normalize_observation_for_key(
            observation, key)

    def denormalize_observation_for_key(self, observation, key):
        return self.robot_observations.denormalize_observation_for_key(
            observation, key)

    def apply_interventions(self, interventions_dict):
        #only will do such an intervention if its a feasible one
        if self.is_initialized():
            old_state = self.get_full_state()
        else:
            old_state = self.get_default_state()
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
            self.set_full_state(np.append(new_joint_positions,
                                          new_joint_velcoities))
        for intervention in interventions_dict:
            if intervention == "joint_velocities" or \
                    intervention == "joint_positions":
                continue
            if "robot_finger" in intervention:
                for sub_intervention_variable in \
                        interventions_dict[intervention]:
                    if sub_intervention_variable == 'color':
                        self.get_pybullet_client().changeVisualShape(
                            self.tri_finger.finger_id,
                            self.link_ids[intervention],
                            rgbaColor=np.append(
                                interventions_dict[intervention]
                                [sub_intervention_variable], 1))
                        if self.enable_goal_image:
                            self.goal_image_instance._p.changeVisualShape(
                                self.tri_finger.finger_id,
                                self.link_ids[intervention],
                                rgbaColor=np.append(
                                    interventions_dict[intervention]
                                    [sub_intervention_variable], 1))
                    elif sub_intervention_variable == 'mass':
                        self.get_pybullet_client().changeDynamics\
                            (self.tri_finger.finger_id,
                             self.link_ids[intervention], mass=
                             interventions_dict[intervention]
                             [sub_intervention_variable])
                    else:
                        raise Exception(
                            "The intervention state variable specified is "
                            "not allowed")

            else:
                raise Exception("The intervention state variable specified is "
                                "not allowed")

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
        for contact in self.tri_finger._p.getContactPoints():
            if (contact[1] == self.tri_finger.finger_id or
                contact[2] == self.tri_finger.finger_id) and \
                    contact[8] < -0.005:
                return False
        return True

    def is_initialized(self):
        if self.latest_full_state is None:
            return False
        else:
            return True

    def is_self_colliding(self):
        for contact in self.tri_finger._p.getContactPoints():
            if contact[1] == self.tri_finger.finger_id and \
                    contact[2] == self.tri_finger.finger_id:
                return True
        return False

    def is_colliding_with_stage(self):
        for contact in self.tri_finger._p.getContactPoints():
            if (contact[1] == self.tri_finger.finger_id and contact[2]
                == self.tri_finger.stage_id) or \
                    (contact[2] == self.tri_finger.finger_id and contact[1]
                     == self.tri_finger.stage_id):
                return True
        return False

    def is_in_contact_with_block(self, block):
        for contact in self.tri_finger._p.getContactPoints():
            if (contact[1] == self.tri_finger.finger_id and
                contact[2] == block.block_id) or \
                    (contact[2] == self.tri_finger.finger_id and
                     contact[1] == block.block_id):
                return True
        return False

    def get_normal_interaction_force_with_block(self, block,
                                                finger_tip_number):
        # TODO: doesnt account for several contacts per body
        if finger_tip_number == 60:
            idx = self.link_ids['robot_finger_60_link_3']
        elif finger_tip_number == 120:
            idx = self.link_ids['robot_finger_120_link_3']
        elif finger_tip_number == 300:
            idx = self.link_ids['robot_finger_300_link_3']
        else:
            raise Exception("finger tip number doesnt exist")

        for contact in self.tri_finger._p.getContactPoints():
            if (contact[1] == self.tri_finger.finger_id and
                contact[2] == block.block_id) or \
                    (contact[2] == self.tri_finger.finger_id
                     and contact[1] == block.block_id):
                return contact[9]*np.array(contact[7])
            for contact in self.tri_finger._p.getContactPoints():
                if (contact[1] == self.tri_finger.finger_id
                    and contact[2] == block.block_id
                    and contact[3] == idx) or  \
                        (contact[2] == self.tri_finger.finger_id
                         and contact[1] == block.block_id
                     and contact[4] == idx):
                    return contact[9] * np.array(contact[7])
        return None

    def get_tip_contact_states(self):
        #TODO: only support open and closed states (should support slipping too)
        contact_tips = [0, 0, 0] #all are open
        for contact in self.tri_finger._p.getContactPoints():
            if contact[1] == self.tri_finger.finger_id:
                if contact[3] == self.link_ids['robot_finger_60_link_3']:
                    contact_tips[0] = 1
                elif contact[3] == self.link_ids['robot_finger_120_link_3']:
                    contact_tips[1] = 1
                elif contact[3] == self.link_ids['robot_finger_300_link_3']:
                    contact_tips[2] = 1
            elif contact[2] == self.tri_finger.finger_id:
                if contact[4] == self.link_ids['robot_finger_60_link_3']:
                    contact_tips[0] = 1
                elif contact[4] == self.link_ids['robot_finger_180_link_3']:
                    contact_tips[1] = 1
                elif contact[4] == self.link_ids['robot_finger_300_link_3']:
                    contact_tips[2] = 1
        return contact_tips
