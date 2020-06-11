#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# The documentation in this code is heavily derived from the official
# documentation of PyBullet at
# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
# among other scattered sources.
# -------------------------------------------------------------------------------------------------
import copy
import os
import numpy as np
import pybullet
import pybullet_data
from causal_rl_bench.envs.pybullet_fingers.action import Action
from causal_rl_bench.envs.pybullet_fingers.observation import Observation
from causal_rl_bench.envs.pybullet_fingers.base_finger import BaseFinger
from causal_rl_bench.envs.pybullet_fingers.camera import Camera


class SimFinger(BaseFinger):
    """
    A simulation environment for the single and the tri-finger robots.
    This environment is based on PyBullet, the official Python wrapper around
    the Bullet-C API.

    Attributes:

        position_gains (array): The kp gains for the pd control of the
            finger(s). Note, this depends on the simulation step size
            and has been set for a simulation rate of 250 Hz.
        velocity_gains (array):The kd gains for the pd control of the
            finger(s). Note, this depends on the simulation step size
            and has been set for a simulation rate of 250 Hz.
        safety_kd (array): The kd gains used for damping the joint motor
            velocities during the safety torque check on the joint motors.
        max_motor_torque (float): The maximum allowable torque that can
            be applied to each motor.
        action_index (int): An index used to enforce the structure of a
            time-series of length 1 for the action in which the application
            of the action precedes (in time) the observation corresponding
            to it. Incremented each time an action is applied.
        observation_index (int): The corresponding index for the observation
            to ensure the same structure as the action time-series for the
            observation time-series (of length 1).

    """

    def __init__(
        self, time_step, enable_visualization
    ):
        """
        Constructor, initializes the physical world we will work in.

        Args:
            time_step (float): It is the time between two simulation steps.
                Defaults to 1./240. Don't set this to be larger than 1./60.
                The gains etc are set according to a time_step of 0.004 s.
            enable_visualization (bool): See BaseFinger.
        """
        # Always enable the simulation for the simulated robot :)
        self.enable_simulation = True

        super().__init__(enable_visualization)

        self.time_step = time_step
        self.position_gains = np.array(
            [10.0, 10.0, 10.0] * 3
        )
        self.velocity_gains = np.array(
            [0.1, 0.3, 0.001] * 3
        )
        self.safety_kd = np.array([0.08, 0.08, 0.04] * 3)
        self.max_motor_torque = 0.36

        self.action_index = -1
        self.observation_index = 0
        self.cameras = []
        self.cameras.append(Camera(camera_position=[0.2496, 0.2458, 0.4190],
                                   camera_orientation=[0.3760, 0.8690,
                                                       -0.2918, -0.1354],
                                   pybullet_client=self._p))
        self.cameras.append(Camera(camera_position=[0.0047, -0.2834, 0.4558],
                                   camera_orientation=[0.9655, -0.0098,
                                                       -0.0065, -0.2603],
                                   pybullet_client=self._p))
        self.cameras.append(Camera(camera_position=[-0.2470, 0.2513, 0.3943],
                                   camera_orientation=[-0.3633, 0.8686,
                                                       -0.3141, 0.1220],
                                   pybullet_client=self._p))
        self.make_physical_world()
        self.disable_velocity_control()
        self.latest_observation = None

    def reset_world(self):
        self.get_pybullet_client().resetSimulation()
        self.get_pybullet_client().setPhysicsEngineParameter(
            deterministicOverlappingPairs=1)
        self.make_physical_world()
        self.disable_velocity_control()

    def Action(self, torque=None, position=None):
        """
        Fill in the fields of the action structure

        Args:
            torque (array): The torques to apply to the motors
            position (array): The absolute angular position to which
                the motors have to be rotated.

        Returns:
            the_action (Action): the action to be applied to the motors
        """
        if torque is None:
            torque = np.array([0.0] * 3 * 3)
        if position is None:
            position = np.array([np.nan] * 3 * 3)

        action = Action(torque, position)

        return action

    def set_real_time_sim(self, switch=0):
        """
        Choose to simulate in real-time or use a desired simulation rate
        Defaults to non-real time

        Args:
            switch (int, 0/1): 1 to set the simulation in real-time.
        """
        self._p.setRealTimeSimulation(switch)

    def make_physical_world(self):
        """
        Set the physical parameters of the world in which the simulation
        will run, and import the models to be simulated
        """
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._p.setTimeStep(self.time_step)

        self._p.loadURDF("plane_transparent.urdf", [0, 0, 0])
        self.import_finger_model()
        self.set_dynamics_properties()
        self.create_stage()

    def set_dynamics_properties(self):
        """
        To change properties of the robot such as its mass, friction, damping,
        maximum joint velocities etc.
        """
        for link_id in self.finger_link_ids:
            self._p.changeDynamics(
                bodyUniqueId=self.finger_id,
                linkIndex=link_id,
                maxJointVelocity=1e3,
                restitution=0.8,
                jointDamping=0.0,
                lateralFriction=0.1,
                spinningFriction=0.1,
                rollingFriction=0.1,
                linearDamping=0.5,
                angularDamping=0.5,
                contactStiffness=0.1,
                contactDamping=0.05,
            )

    def create_stage(self):
        """Create the stage (table and boundary).

        Args:
        """

        def mesh_path(filename):
            return os.path.join(
                self.robot_properties_path, "meshes", "stl", filename
            )

        table_colour = (0.31, 0.27, 0.25, 1.0)
        high_border_colour = (0.95, 0.95, 0.95, 1.0)
        floor_id = self._p.createCollisionShape(
            shapeType=pybullet.GEOM_MESH, fileName=mesh_path("trifinger_table_without_border.stl"),
            flags=0
        )
        # pybullet.GEOM_FORCE_CONCAVE_TRIMESH
        obj = self._p.createMultiBody(
            baseCollisionShapeIndex=floor_id,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0.01],
            baseOrientation=[0, 0, 0, 1],
        )

        # set colour
        self._p.changeVisualShape(obj, -1, rgbaColor=table_colour)

        stage_id = self._p.createCollisionShape(
            shapeType=pybullet.GEOM_MESH, fileName=mesh_path("high_table_boundary.stl"),
            flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        # pybullet.GEOM_FORCE_CONCAVE_TRIMESH
        obj = self._p.createMultiBody(
            baseCollisionShapeIndex=stage_id,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0.01],
            baseOrientation=[0, 0, 0, 1],
        )

        # set colour
        self._p.changeVisualShape(obj, -1, rgbaColor=high_border_colour)
        return

    def disable_velocity_control(self):
        """
        To disable the high friction velocity motors created by
        default at all revolute and prismatic joints while loading them from
        the urdf.
        """

        self._p.setJointMotorControlArray(
            bodyUniqueId=self.finger_id,
            jointIndices=self.revolute_joint_ids,
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=[0] * len(self.revolute_joint_ids),
            forces=[0] * len(self.revolute_joint_ids),
        )

    def _set_desired_action(self, desired_action):
        """Set the given action after performing safety checks.

        Args:
            desired_action (Action): Joint positions or torques or both

        Returns:
            applied_action:  The action that is actually applied after
                performing the safety checks.
        """
        # copy the action in a way that works for both Action and
        # robot_interfaces.(tri)finger.Action.  Note that a simple
        # copy.copy(desired_action) does **not** work for robot_interfaces
        # actions!
        applied_action = type(desired_action)(
            copy.copy(desired_action.torque),
            copy.copy(desired_action.position),
        )

        def set_gains(gains, defaults):
            """Replace NaN entries in gains with values from defaults."""
            mask = np.isnan(gains)
            output = copy.copy(gains)
            output[mask] = defaults[mask]
            return output

        applied_action.position_kp = set_gains(
            desired_action.position_kp, self.position_gains
        )
        applied_action.position_kd = set_gains(
            desired_action.position_kd, self.velocity_gains
        )

        torque_command = np.asarray(copy.copy(desired_action.torque))
        if not np.isnan(desired_action.position).all():
            torque_command += np.array(
                self.compute_pd_control_torques(
                    desired_action.position,
                    applied_action.position_kp,
                    applied_action.position_kd,
                )
            )

        applied_action.torque = self._set_motor_torques(
            torque_command.tolist()
        )

        return applied_action

    def append_desired_action(self, action):
        """
        Pass an action on which safety checks
        will be performed and then the action will be applied to the motors.

        Args:
            action (Action): Joint positions or torques or both

        Returns:
            self.action_index (int): The current time-index at which the action
                was applied.
        """
        # if self.action_index >= self.observation_index:
        #     raise Exception(
        #         "You have to call get_observation after each"
        #         "append_desired_action."
        #     )

        self._set_desired_action(action)

        self.action_index = self.action_index + 1
        return self.action_index

    def _get_latest_observation(self, update_images):
        """Get observation of the current state.

        Returns:
            observation (Observation): the joint positions, velocities, and
                torques of the joints.
        """
        observation = Observation()
        current_joint_states = self._p.getJointStates(
            self.finger_id, self.revolute_joint_ids
        )

        observation.position = np.array(
            [joint[0] for joint in current_joint_states]
        )
        observation.velocity = np.array(
            [joint[1] for joint in current_joint_states]
        )
        observation.torque = np.array(
            [joint[3] for joint in current_joint_states]
        )
        if update_images:
            observation.camera_60 = self.cameras[0].get_image()
            observation.camera_180 = self.cameras[1].get_image()
            observation.camera_300 = self.cameras[2].get_image()
        else:
            observation.camera_60 = self.latest_observation.camera_60
            observation.camera_180 = self.latest_observation.camera_180
            observation.camera_300 = self.latest_observation.camera_300

        return observation

    def get_observation(self, time_index, update_images=False):
        """
        Get the observation at the time of
        applying the action, so the observation actually corresponds
        to the state of the environment due to the application of the
        previous action.

        This method steps the simulation!

        Args:
            time_index (int): the time index at which the observation is
                needed. This can only be the current time-index (so same as the
                action_index)

        Returns:
            observation (Observation): the joint positions, velocities, and
                torques of the joints.

        Raises:
            Exception if the observation at any other time index than the one
            at which the action is applied, is queried for.
        """
        # if not time_index == self.action_index:
        #     raise Exception(
        #         "currently you can only get the latest" "observation"
        #     )
        #
        # assert (
        #     self.observation_index == self.action_index
        # ), "observation_index {} != action_index {}".format(
        #     self.observation_index, self.action_index
        # )

        observation = self._get_latest_observation(update_images)
        self.latest_observation = observation
        self.observation_index = self.observation_index + 1
        return observation

    def step_simulation(self):
        self._step_simulation()
        return

    def compute_pd_control_torques(self, joint_positions, kp=None, kd=None):
        """
        Compute torque command to reach given target position using a PD
        controller.

        Args:
            joint_positions (array-like, shape=(n,)):  Desired joint positions.
            kp (array-like, shape=(n,)): P-gains, one for each joint.
            kd (array-like, shape=(n,)): D-gains, one for each joint.

        Returns:
            List of torques to be sent to the joints of the finger in order to
            reach the specified joint_positions.
        """
        if kp is None:
            kp = self.position_gains
        if kd is None:
            kd = self.velocity_gains

        current_joint_states = self._p.getJointStates(
            self.finger_id, self.revolute_joint_ids
        )
        current_position = np.array(
            [joint[0] for joint in current_joint_states]
        )
        current_velocity = np.array(
            [joint[1] for joint in current_joint_states]
        )

        position_error = joint_positions - current_position

        position_feedback = np.asarray(kp) * position_error
        velocity_feedback = np.asarray(kd) * current_velocity

        joint_torques = position_feedback - velocity_feedback

        # set nan entries to zero (nans occur on joints for which the target
        # position was set to nan)
        joint_torques[np.isnan(joint_torques)] = 0.0

        return joint_torques.tolist()

    def pybullet_inverse_kinematics(self, desired_tip_positions, rest_pose):
        """
        Compute the joint angular positions needed to get to reach the block.

        WARNING: pybullet's inverse kinematics seem to be very inaccurate! (or
        we are somehow using it wrongly...)

        Args:
            finger (SimFinger): a SimFinger object
            desired_tip_position (list of floats): xyz target position for
                each finger tip.

        Returns:
            joint_pos (list of floats): The angular positions to be applid at
                the joints to reach the desired_tip_position
        """
        joint_pos = np.zeros([9])
        finger_tip_ids = self.finger_tip_ids
        final_joint_pose = self._p.calculateInverseKinematics2(self.finger_id,
                                                               [finger_tip_ids[0], finger_tip_ids[1],
                                                                finger_tip_ids[2]],
                                                               [desired_tip_positions[0:3], desired_tip_positions[3:6],
                                                                desired_tip_positions[6:]],
                                                               solver=self._p.IK_DLS,
                                                               currentPositions=rest_pose)
        joint_pos[:3] = final_joint_pose[:3]
        final_joint_pose = self._p.calculateInverseKinematics2(self.finger_id,
                                                               [finger_tip_ids[1], finger_tip_ids[0],
                                                                finger_tip_ids[2]],
                                                               [desired_tip_positions[3:6], desired_tip_positions[0:3],
                                                                desired_tip_positions[6:]],
                                                               solver=self._p.IK_DLS,
                                                               currentPositions=rest_pose)
        joint_pos[3:6] = final_joint_pose[3:6]
        final_joint_pose = self._p.calculateInverseKinematics2(self.finger_id,
                                                               [finger_tip_ids[2], finger_tip_ids[0],
                                                                finger_tip_ids[1]],
                                                               [desired_tip_positions[6:], desired_tip_positions[0:3],
                                                                desired_tip_positions[3:6]],
                                                               solver=self._p.IK_DLS,
                                                               currentPositions=rest_pose)
        joint_pos[6:] = final_joint_pose[6:]
        if np.isnan(joint_pos).any():
            joint_pos = rest_pose
        return joint_pos

    def _set_motor_torques(self, desired_torque_commands):
        """
        Send torque commands to the motors.

        Args:
            desired_torque_commands (list of floats): The desired torques to be
                applied to the motors.  The torques that are actually applied
                may differ as some safety checks are applied.  See return
                value.

        Returns:
            List of torques that is actually set after applying safety checks.

        """
        torque_commands = self._safety_torque_check(desired_torque_commands)

        self._p.setJointMotorControlArray(
            bodyUniqueId=self.finger_id,
            jointIndices=self.revolute_joint_ids,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=torque_commands,
        )

        return torque_commands

    def _safety_torque_check(self, desired_torques):
        """
        Perform a check on the torques being sent to be applied to
        the motors so that they do not exceed the safety torque limit

        Args:
            desired_torques (list of floats): The torques desired to be
                applied to the motors

        Returns:
            applied_torques (list of floats): The torques that can be actually
                applied to the motors (and will be applied)
        """
        applied_torques = np.clip(
            np.asarray(desired_torques),
            -self.max_motor_torque,
            +self.max_motor_torque,
        )

        current_joint_states = self._p.getJointStates(
            self.finger_id, self.revolute_joint_ids
        )
        current_velocity = np.array(
            [joint[1] for joint in current_joint_states]
        )
        applied_torques -= self.safety_kd * current_velocity

        applied_torques = list(
            np.clip(
                np.asarray(applied_torques),
                -self.max_motor_torque,
                +self.max_motor_torque,
            )
        )

        return applied_torques

    def _step_simulation(self):
        """
        Step the simulation to go to the next world state.
        """
        self._p.stepSimulation()

    def reset_finger(self, joint_positions, joint_velocities=None):
        """
        Reset the finger(s) to some random position (sampled in the joint
        space) and step the robot with this random position

        Args:
            joint_positions (array-like):  Angular position for each joint.  If
                None, a random position is sampled.
            joint_velocities (array-like): Angular velocity for each joint.  If
                None, its set to 0.
        """
        if joint_velocities is None:
            for i, joint_id in enumerate(self.revolute_joint_ids):
                self._p.resetJointState(
                    self.finger_id, joint_id, joint_positions[i]
                )
        else:
            for i, joint_id in enumerate(self.revolute_joint_ids):
                self._p.resetJointState(
                    self.finger_id, joint_id, joint_positions[i],
                    joint_velocities[i]
                )
        t = self.append_desired_action(self.Action(position=joint_positions))
        self.step_simulation()
        return self.get_observation(t, update_images=True)

    def finger_intervention(self, joint_positions, joint_velocities=None):
        """
        Reset the finger(s) to some random position (sampled in the joint
        space) and step the robot with this random position

        Args:
            joint_positions (array-like):  Angular position for each joint.  If
                None, a random position is sampled.
            joint_velocities (array-like): Angular velocity for each joint.  If
                None, its set to 0.
        """
        if joint_velocities is None:
            for i, joint_id in enumerate(self.revolute_joint_ids):
                self._p.resetJointState(
                    self.finger_id, joint_id, joint_positions[i]
                )
        else:
            for i, joint_id in enumerate(self.revolute_joint_ids):
                self._p.resetJointState(
                    self.finger_id, joint_id, joint_positions[i],
                    joint_velocities[i]
                )
        t = self.append_desired_action(self.Action(position=joint_positions))
        return self.get_observation(t, update_images=True)
