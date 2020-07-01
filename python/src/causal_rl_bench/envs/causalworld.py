import numpy as np
import gym
import pybullet
import pybullet_data
import os
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.utils.env_utils import combine_spaces
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.robot.camera import Camera
from causal_rl_bench.configs.world_constants import WorldConstants
import copy
import pkgutil
from causal_rl_bench.envs.robot.pinocchio_utils import PinocchioUtils


class CausalWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, task=None, skip_frame=10,
                 enable_visualization=False, seed=0,
                 action_mode="joint_positions", observation_mode="structured",
                 normalize_actions=True, normalize_observations=True,
                 max_episode_length=None, data_recorder=None):
        """

        :param task:
        :param skip_frame:
        :param enable_visualization:
        :param seed:
        :param action_mode:
        :param observation_mode:
        :param normalize_actions:
        :param normalize_observations:
        :param max_episode_length:
        :param data_recorder:
        :param enable_goal_image:
        :param kwargs:
        """
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._enable_visualization = enable_visualization
        self.seed(seed)
        self._simulation_time = 1. / 250
        self._skip_frame = skip_frame
        self.dt = self._simulation_time * self._skip_frame
        self._pybullet_client_w_o_goal_id = None
        self._pybullet_client_w_goal_id = None
        self._pybullet_client_full_id = None
        self._revolute_joint_ids = None
        self._instantiate_pybullet()
        self.link_name_to_index = None
        self._robot_properties_path = os.path.join(
            os.path.
                dirname(__file__), "../../../assets/robot_properties_fingers"
        )
        self._finger_urdf_path = os.path.join(
            self._robot_properties_path, "urdf", "trifinger.urdf"
        )
        self._create_world(initialize_goal_image=True)
        self._pinocchio_utils = PinocchioUtils(self._finger_urdf_path)
        self._tool_cameras = None
        self._goal_cameras = None
        if observation_mode == 'cameras':
            self._tool_cameras = []
            self._tool_cameras.append(
                Camera(camera_position=[0.2496, 0.2458, 0.4190],
                       camera_orientation=[0.3760, 0.8690,
                                           -0.2918, -0.1354],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._tool_cameras.append(
                Camera(camera_position=[0.0047, -0.2834, 0.4558],
                       camera_orientation=[0.9655, -0.0098,
                                           -0.0065, -0.2603],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._tool_cameras.append(
                Camera(camera_position=[-0.2470, 0.2513, 0.3943],
                       camera_orientation=[-0.3633, 0.8686,
                                           -0.3141, 0.1220],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._goal_cameras = []
            self._goal_cameras.append(
                Camera(camera_position=[0.2496, 0.2458, 0.4190],
                       camera_orientation=[0.3760, 0.8690,
                                           -0.2918, -0.1354],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
            self._goal_cameras.append(
                Camera(camera_position=[0.0047, -0.2834, 0.4558],
                       camera_orientation=[0.9655, -0.0098,
                                           -0.0065, -0.2603],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
            self._goal_cameras.append(
                Camera(camera_position=[-0.2470, 0.2513, 0.3943],
                       camera_orientation=[-0.3633, 0.8686,
                                           -0.3141, 0.1220],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
        self._robot = TriFingerRobot(action_mode=action_mode,
                                     observation_mode=observation_mode,
                                     skip_frame=skip_frame,
                                     normalize_actions=normalize_actions,
                                     normalize_observations=
                                     normalize_observations,
                                     simulation_time=self._simulation_time,
                                     pybullet_client_full_id=
                                     self._pybullet_client_full_id,
                                     pybullet_client_w_goal_id=
                                     self._pybullet_client_w_goal_id,
                                     pybullet_client_w_o_goal_id=
                                     self._pybullet_client_w_o_goal_id,
                                     revolute_joint_ids=
                                     self._revolute_joint_ids,
                                     finger_tip_ids=self.finger_tip_ids,
                                     cameras=self._tool_cameras,
                                     pinocchio_utils=self._pinocchio_utils)
        self._stage = Stage(observation_mode=observation_mode,
                            normalize_observations=normalize_observations,
                            pybullet_client_full_id=
                            self._pybullet_client_full_id,
                            pybullet_client_w_goal_id=
                            self._pybullet_client_w_goal_id,
                            pybullet_client_w_o_goal_id=
                            self._pybullet_client_w_o_goal_id,
                            cameras=self._goal_cameras)
        gym.Env.__init__(self)
        if task is None:
            self._task = task_generator("reaching")
        else:
            self._task = task
        self._task.init_task(self._robot, self._stage, max_episode_length,
                             self._create_world)
        if max_episode_length is None:
            max_episode_length = int(task.get_default_max_episode_length() /
                                     self.dt)
        self._max_episode_length = max_episode_length
        self._reset_observations_space()
        self.action_space = self._robot.get_action_spaces()

        self.metadata['video.frames_per_second'] = \
            (1 / self._simulation_time) / self._skip_frame
        # TODO: verify spaces here
        #TODO: we postpone this function for now
        self._setup_viewing_camera()
        self._normalize_actions = normalize_actions
        self._normalize_observations = normalize_observations
        self._episode_length = 0
        self._data_recorder = data_recorder
        self._wrappers_dict = dict()
        self._tracker = Tracker(task=self._task,
                                world_params=self.get_world_params())
        self._scale_reward_by_dt = True
        self._disabled_actions = False
        #TODO: I am not sure if this reset is necassary, TO BE CONFIRMED
        # self.reset()
        return

    def are_actions_normalized(self):
        return self._normalize_actions

    def are_observations_normalized(self):
        return self._normalize_observations

    def _reset_observations_space(self):
        """

        :return:
        """
        if self._observation_mode == "cameras" and self.observation_space is None:
            self._stage.select_observations(["goal_image"])
            self.observation_space = combine_spaces(
                self._robot.get_observation_spaces(),
                self._stage.get_observation_spaces())
        elif self._observation_mode == "cameras" and self.observation_space is not None:
            return
        else:
            self._robot.select_observations(self._task._task_robot_observation_keys)
            self._stage.select_observations(self._task._task_stage_observation_keys)
            self.observation_space = \
                combine_spaces(self._robot.get_observation_spaces(),
                               self._stage.get_observation_spaces())
        return

    def step(self, action):
        """

        :param action:
        :return:
        """
        self._episode_length += 1
        if not self._disabled_actions:
            self._robot.apply_action(action)
        if self._observation_mode == "cameras":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            observation = np.concatenate((current_images, goal_images),
                                         axis=0)
        else:
            observation = self._task.filter_structured_observations()
        reward = self._task.get_reward()
        info = self._task.get_info()
        if self._scale_reward_by_dt:
            reward *= self.dt
        done = self._is_done()
        if self._data_recorder:
            self._data_recorder.append(robot_action=action,
                                       observation=observation,
                                       reward=reward,
                                       done=done,
                                       info=info,
                                       timestamp=self._episode_length *
                                                  self._skip_frame *
                                                  self._simulation_time)

        return observation, reward, done, info

    def reset_default_state(self):
        """

        :return:
        """
        self._task.reset_default_state()
        return

    def sample_new_goal(self, training=True, level=None):
        """

        :param training:
        :param level:
        :return:
        """
        return self._task.sample_new_goal(training, level)

    def _disable_actions(self):
        self._disabled_actions = True

    def _add_data_recorder(self, data_recorder):
        self._data_recorder = data_recorder

    def seed(self, seed=None):
        """

        :param seed:
        :return:
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, interventions_dict=None):
        """

        :param interventions_dict:
        :return:
        """
        self._tracker.add_episode_experience(self._episode_length)
        self._episode_length = 0
        if interventions_dict is not None:
            interventions_dict = copy.deepcopy(interventions_dict)
            self._tracker.do_intervention(self._task, interventions_dict)
        success_signal, interventions_info, reset_observation_space_signal = \
            self._task.reset_task(interventions_dict)
        if reset_observation_space_signal:
            self._reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self._tracker.add_invalid_intervention(interventions_info)
        # TODO: make sure that stage observations returned are up to date
        if self._data_recorder:
            self._data_recorder.new_episode(self.get_state(),
                                            task_name=
                                            self._task._task_name,
                                            task_params=
                                            self._task.get_task_params(),
                                            world_params=
                                            self.get_world_params())
        if self._observation_mode == "cameras":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            return np.concatenate((current_images, goal_images), axis=0)
        else:
            return self._task.filter_structured_observations()

    def close(self):
        """

        :return:
        """
        if self._data_recorder:
            self._data_recorder.save()
        self._robot.close()

    def _get_tracker(self):
        return self._tracker

    def _is_done(self):
        if self._episode_length > self._max_episode_length:
            return True
        else:
            return self._task.is_done()

    def do_single_random_intervention(self):
        """

        :return:
        """
        success_signal, interventions_info, interventions_dict, reset_observation_space_signal = \
            self._task.do_single_random_intervention()
        if reset_observation_space_signal:
            self._reset_observations_space()
        if len(interventions_dict) > 0:
            self._tracker.do_intervention(self._task, interventions_dict)
            if success_signal is not None:
                if not success_signal:
                    self._tracker.add_invalid_intervention(interventions_info)
        return interventions_dict, success_signal

    def do_intervention(self, interventions_dict,
                        check_bounds=None):
        """

        :param interventions_dict:
        :param check_bounds:
        :return:
        """
        success_signal, interventions_info, reset_observation_space_signal = \
            self._task.do_intervention(interventions_dict,
                                       check_bounds=check_bounds)
        self._tracker.do_intervention(self._task, interventions_dict)
        if reset_observation_space_signal:
            self._reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                self._tracker.add_invalid_intervention(interventions_info)
        return success_signal

    def get_state(self):
        """
        Note: Setting state and getting state doesnt work when there is an intermediate intervention
        :return:
        """
        state = dict()
        state['pybullet_state'] = self._task._save_pybullet_state()
        state['control_index'] = self._robot._control_index
        return state
        # return self._task.save_state()

    def set_state(self, new_full_state):
        """

        :param new_full_state:
        :return:
        """
        self._task._restore_pybullet_state(new_full_state['pybullet_state'])
        self._robot._control_index = new_full_state['control_index']
        self._robot.update_latest_full_state()
        # self._task.restore_state(new_full_state)
        return

    def render(self, mode="human"):
        """

        :param mode:
        :return:
        """
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        (_, _, px, _, _) = pybullet.getCameraImage(
            width=self._render_width, height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _setup_viewing_camera(self):
        """

        :return:
        """
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        self._cam_dist = 1
        self._cam_yaw = 0
        self._cam_pitch = -60
        self._render_width = 320
        self._render_height = 240
        base_pos = [0, 0, 0]

        self.view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=client)
        self.proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0,
            physicsClientId=client
        )

    def get_current_state_variables(self):
        """

        :return:
        """
        return self._task.get_current_state_variables()

    def get_world_params(self):
        """

        :return:
        """
        world_params = dict()
        world_params["skip_frame"] = self._robot.get_skip_frame()
        world_params["action_mode"] = self._robot.get_action_mode()
        world_params["observation_mode"] = self._robot.get_observation_mode()
        world_params["normalize_actions"] = \
            self._robot._robot_actions.is_normalized()
        world_params["normalize_observations"] = \
            self._robot._robot_observations.is_normalized()
        world_params["max_episode_length"] = self._max_episode_length
        world_params["simulation_time"] = self._simulation_time
        world_params["wrappers"] = self._wrappers_dict
        return world_params

    def _add_wrapper_info(self, wrapper_dict):
        """

        :param wrapper_dict:
        :return:
        """
        self._wrappers_dict.update(wrapper_dict)
        return

    def save_world(self, log_relative_path):
        """

        :param log_relative_path:
        :return:
        """
        if not os.path.exists(log_relative_path):
            os.makedirs(log_relative_path)
        tracker_path = os.path.join(log_relative_path, 'tracker')
        tracker = self._get_tracker()
        tracker.save(file_path=tracker_path)
        return

    def is_in_training_mode(self):
        """

        :return:
        """
        return self._task.is_in_training_mode()

    def get_joint_positions_lower_bound(self):
        return self._robot._robot_actions.\
            joint_positions_lower_bounds

    def get_action_mode(self):
        return self._action_mode

    def set_action_mode(self, action_mode):
        self._action_mode = action_mode
        self._robot.set_action_mode(action_mode)

    def get_robot(self):
        return self._robot

    def get_task(self):
        return self._task

    def get_stage(self):
        return self._stage

    def get_tracker(self):
        return self._tracker

    def _create_world(self, initialize_goal_image=False):
        """
        This function loads the urdfs of the robot in all the pybullet clients
        :return:
        """
        self._reset_world()
        finger_base_position = [0, 0, 0.0]
        finger_base_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        if initialize_goal_image:
            client_list = [self._pybullet_client_w_o_goal_id,
                           self._pybullet_client_w_goal_id,
                           self._pybullet_client_full_id]
        else:
            client_list = [self._pybullet_client_w_o_goal_id,
                           self._pybullet_client_full_id]

        for client in client_list:
            if client is not None:
                pybullet.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                                 physicsClientId=client)
                pybullet.setGravity(0, 0, -9.81,
                                  physicsClientId=client)
                pybullet.setTimeStep(self._simulation_time,
                                     physicsClientId=client)
                pybullet.loadURDF("plane_transparent.urdf", [0, 0, 0],
                                physicsClientId=client)
                pybullet.loadURDF(
                    fileName=self._finger_urdf_path,
                    basePosition=finger_base_position,
                    baseOrientation=finger_base_orientation,
                    useFixedBase=1,
                    flags=(
                            pybullet.URDF_USE_INERTIA_FROM_FILE
                            | pybullet.URDF_USE_SELF_COLLISION
                    ),
                    physicsClientId=client
                )
                if self.link_name_to_index is None:
                    self.link_name_to_index = {
                        pybullet.getBodyInfo(WorldConstants.ROBOT_ID,
                                             physicsClientId=client)[0].decode(
                            "UTF-8"): -1,
                    }
                    for joint_idx in range(
                            pybullet.getNumJoints(WorldConstants.ROBOT_ID,
                                                  physicsClientId=client)):
                        link_name = pybullet.getJointInfo(
                            WorldConstants.ROBOT_ID, joint_idx,
                            physicsClientId=client)[
                            12
                        ].decode("UTF-8")
                        self.link_name_to_index[link_name] = joint_idx

                    self._revolute_joint_ids = [
                        self.link_name_to_index[name] for name in
                        WorldConstants.JOINT_NAMES
                    ]
                    self.finger_tip_ids = [
                        self.link_name_to_index[name] for name in
                        WorldConstants.TIP_LINK_NAMES
                    ]
                    # joint and link indices are the same in pybullet
                    # TODO do we even need this variable?
                    self.finger_link_ids = self._revolute_joint_ids
                    self.last_joint_position = [0] * len(self._revolute_joint_ids)
                for link_id in self.finger_link_ids:
                    pybullet.changeDynamics(
                                    bodyUniqueId=WorldConstants.ROBOT_ID,
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
                                    physicsClientId=client
                                )
                self._create_stage(client)
        return

    def _reset_world(self):
        if self._pybullet_client_full_id is not None:
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_full_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_full_id)
        if self._pybullet_client_w_o_goal_id is not None:
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_o_goal_id)
        return

    def _create_stage(self, pybullet_client):
        """Create the stage (table and boundary).

        Args:
        """

        def mesh_path(filename):
            return os.path.join(
                self._robot_properties_path, "meshes", "stl", filename
            )

        table_colour = (0.31, 0.27, 0.25, 1.0)
        high_border_colour = (0.95, 0.95, 0.95, 1.0)
        floor_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=mesh_path("trifinger_table_without_border.stl"),
            flags=0,
            physicsClientId=pybullet_client
        )
        obj = pybullet.createMultiBody(
            baseCollisionShapeIndex=floor_id,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0.01],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=pybullet_client
        )
        pybullet.changeVisualShape(obj, -1, rgbaColor=table_colour,
                                   physicsClientId=pybullet_client)

        stage_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=mesh_path("high_table_boundary.stl"),
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=pybullet_client
        )
        obj = pybullet.createMultiBody(
            baseCollisionShapeIndex=stage_id,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0.01],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=pybullet_client
        )
        pybullet.changeVisualShape(obj, -1, rgbaColor=high_border_colour,
                                   physicsClientId=pybullet_client)
        return

    def _instantiate_pybullet(self):
        """
        This function is used for instantiating all pybullet instances used for
        the current simulation
        :return:
        """
        if self._observation_mode == 'cameras':
            self._pybullet_client_w_o_goal_id = pybullet.connect(
                pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI, 0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_w_o_goal_id
            )
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            self._pybullet_client_w_goal_id = pybullet.connect(
                pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI, 0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_w_goal_id
            )
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_goal_id)
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(
                    pybullet.GUI)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_GUI, 0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.resetSimulation(
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.setPhysicsEngineParameter(
                    deterministicOverlappingPairs=1,
                    physicsClientId=self._pybullet_client_full_id)
        else:
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(
                    pybullet.GUI)
            else:
                self._pybullet_client_full_id = pybullet.connect(
                    pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI, 0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_full_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_full_id)
        return
