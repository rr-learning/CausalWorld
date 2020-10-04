import numpy as np
import gym
import pybullet
import pybullet_data
import os
from causal_world.envs.robot.trifinger import TriFingerRobot
from causal_world.envs.scene.stage import Stage
from causal_world.loggers.tracker import Tracker
from causal_world.utils.env_utils import combine_spaces
from causal_world.task_generators.task import generate_task
from causal_world.envs.robot.camera import Camera
from causal_world.configs.world_constants import WorldConstants
import copy
import logging
logging.getLogger().setLevel(logging.INFO)


class CausalWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 task=None,
                 skip_frame=10,
                 enable_visualization=False,
                 seed=0,
                 action_mode="joint_positions",
                 observation_mode="structured",
                 normalize_actions=True,
                 normalize_observations=True,
                 max_episode_length=None,
                 data_recorder=None,
                 camera_indicies=np.array([0, 1, 2]),
                 wrappers=None):
        """
        The causal world encapsulates the environment of the agent, where you
        can perform actions, intervene, reset the state..etc

        :param task: (causal_world.BaseTask) this is the task produced by one of
                                            the available task generators or the
                                            custom task created.
        :param skip_frame: (int) the low level controller is running @250Hz
                           which corresponds to skip frame of 1, a skip frame
                           of 250 corresponds to frequency of 1Hz
        :param enable_visualization: (bool) this is a boolean which indicates
                                     if a GUI is enabled or the environment
                                     should operate in a headless mode.
        :param seed: (int) this is the random seed used in the environment.
        :param action_mode: (str) action_modes available are "joint_positions",
                                     "end_effector_positions" which is non
                                     deterministic at the moment since it uses
                                     IK of pybullet and lastly "joint_torques".
        :param observation_mode: (str) observation modes available are
                                          "structured" or "pixel" modes.
                                          The structured observations are
                                          specified in the task generator
                                          itself. For the "pixel" mode
                                          you will get maximum 6 images
                                          concatenated where the first half
                                          are the current images rendered in
                                          the platform and the second half are
                                          the goal images rendered from the
                                          same point of view.
        :param normalize_actions: (bool) this is a boolean which specifies
                                         whether the actions passed to the step
                                         function are normalized or not.
        :param normalize_observations: (bool) this is a boolean which specifies
                                              whether the observations returned
                                              should be normalized or not.
        :param max_episode_length: (int) it specifies what is the episode length
                                         of the task, by default this will be
                                         calculated according to how many
                                         objects are in the arena
                                         (5 secs * number of objects).
        :param data_recorder: (causal_world.DataRecorder) passed only when
                                                            you want to log and
                                                            record the episodes
                                                            which can be used
                                                            further in imitation
                                                            learning for instance.
        :param camera_indicies: (list) maximum of 3 elements where each element
                                       is from 0 to , specifies which cameras
                                       to return in the observations and the
                                       order as well.
        :param wrappers: (causal_world.wrappers) should not be used for now.
        """
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._enable_visualization = enable_visualization
        self.seed(seed)
        self._simulation_time = 1. / 250
        self._camera_indicies = np.array(camera_indicies)
        self._skip_frame = skip_frame
        self.dt = self._simulation_time * self._skip_frame
        self._pybullet_client_w_o_goal_id = None
        self._pybullet_client_w_goal_id = None
        self._pybullet_client_full_id = None
        self._revolute_joint_ids = None
        self._instantiate_pybullet()
        self.link_name_to_index = None
        self._robot_properties_path = os.path.join(
            os.path.dirname(__file__),
            "../assets/robot_properties_fingers")
        self._finger_urdf_path = os.path.join(self._robot_properties_path,
                                              "urdf", "trifinger_edu.urdf")
        self._create_world(initialize_goal_image=True)
        self._tool_cameras = None
        self._goal_cameras = None
        if observation_mode == 'pixel':
            self._tool_cameras = []
            self._tool_cameras.append(
                Camera(camera_position=[0.2496, 0.2458, 0.58],
                       camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._tool_cameras.append(
                Camera(camera_position=[0.0047, -0.2834, 0.58],
                       camera_orientation=[0.9655, -0.0098, -0.0065, -0.2603],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._tool_cameras.append(
                Camera(camera_position=[-0.2470, 0.2513, 0.50],
                       camera_orientation=[-0.3633, 0.8686, -0.3141, 0.1220],
                       pybullet_client_id=self._pybullet_client_w_o_goal_id))
            self._goal_cameras = []
            self._goal_cameras.append(
                Camera(camera_position=[0.2496, 0.2458, 0.58],
                       camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
            self._goal_cameras.append(
                Camera(camera_position=[0.0047, -0.2834, 0.58],
                       camera_orientation=[0.9655, -0.0098, -0.0065, -0.2603],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
            self._goal_cameras.append(
                Camera(camera_position=[-0.2470, 0.2513, 0.50],
                       camera_orientation=[-0.3633, 0.8686, -0.3141, 0.1220],
                       pybullet_client_id=self._pybullet_client_w_goal_id))
        self._robot = TriFingerRobot(
            action_mode=action_mode,
            observation_mode=observation_mode,
            skip_frame=skip_frame,
            normalize_actions=normalize_actions,
            normalize_observations=normalize_observations,
            simulation_time=self._simulation_time,
            pybullet_client_full_id=self._pybullet_client_full_id,
            pybullet_client_w_goal_id=self._pybullet_client_w_goal_id,
            pybullet_client_w_o_goal_id=self._pybullet_client_w_o_goal_id,
            revolute_joint_ids=self._revolute_joint_ids,
            finger_tip_ids=self.finger_tip_ids,
            cameras=self._tool_cameras,
            camera_indicies=self._camera_indicies)
        self._stage = Stage(
            observation_mode=observation_mode,
            normalize_observations=normalize_observations,
            pybullet_client_full_id=self._pybullet_client_full_id,
            pybullet_client_w_goal_id=self._pybullet_client_w_goal_id,
            pybullet_client_w_o_goal_id=self._pybullet_client_w_o_goal_id,
            cameras=self._goal_cameras,
            camera_indicies=self._camera_indicies)
        gym.Env.__init__(self)
        if task is None:
            self._task = generate_task("reaching")
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
        return

    def set_skip_frame(self, new_skip_frame):
        """

        :param new_skip_frame: (int) the new skip frame of the environment.
        :return:
        """
        self._skip_frame = new_skip_frame
        self.dt = self._simulation_time * self._skip_frame
        self.metadata['video.frames_per_second'] = \
            (1 / self._simulation_time) / self._skip_frame
        self._robot._skip_frame = new_skip_frame
        self._robot._dt = self._simulation_time * \
                          self._robot._skip_frame
        return

    def expose_potential_partial_solution(self):
        """
        Adds the partial solution field in the info dict returned after stepping
        Through the environment.

        :return: None
        """
        self._task.expose_potential_partial_solution()
        return

    def add_ground_truth_state_to_info(self):
        """
        Adds the ground truth state variables in the info dict returned after
        stepping through the environment, can be used in representation
        learning.

        :return: None
        """
        self._task.add_ground_truth_state_to_info()
        return

    def are_actions_normalized(self):
        """
        :return: (bool) whether the actions are normalized or not.
        """
        return self._normalize_actions

    def are_observations_normalized(self):
        """
        :return: (bool) whether the observations are normalized or not.
        """
        return self._normalize_observations

    def _reset_observations_space(self):
        if self._observation_mode == "pixel" and self.observation_space is None:
            self._stage.select_observations(["goal_image"])
            self.observation_space = combine_spaces(
                self._robot.get_observation_spaces(),
                self._stage.get_observation_spaces())
        elif self._observation_mode == "pixel" and self.observation_space is not None:
            return
        else:
            self._robot.select_observations(
                self._task._task_robot_observation_keys)
            self._stage.select_observations(
                self._task._task_stage_observation_keys)
            self.observation_space = \
                combine_spaces(self._robot.get_observation_spaces(),
                               self._stage.get_observation_spaces())
        return

    def step(self, action):
        """
        Used to step through the enviroment.

        :param action: (nd.array) specifies which action should be taken by
                                  the robot, should follow the same action
                                  mode specified.

        :return: (nd.array) specifies the observations returned after stepping
                            through the environment. Again, it follows the
                            observation_mode specified.
        """
        self._episode_length += 1
        if not self._disabled_actions:
            self._robot.apply_action(action)
        if self._observation_mode == "pixel":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            observation = np.concatenate((current_images, goal_images), axis=0)
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
        Used to reset the environment starting state to the default starting
        state specified by the task generator. Can be used if an intervention
        on the starting state was performed and its needed to go back to the
        default starting state that was specified before.

        :return: None
        """
        self._task.reset_default_state()
        return

    def sample_new_goal(self, level=None):
        """
        Used to sample a new goal in the task subspace, the environment is
        operating in, so goals are still restricted depends on the task
        generator that was used.

        :param level: (int) this is not used at the moment.

        :return: None
        """
        return self._task.sample_new_goal(level)

    def _disable_actions(self):
        """
        Disables the actions to be performed.
        :return:
        """
        self._disabled_actions = True
        return

    def _add_data_recorder(self, data_recorder):
        """
        Adds a data recorder to the environment.
        :param data_recorder: (causal_world.loggers.DataRecorder) data recorder of the system.
        :return:
        """
        self._data_recorder = data_recorder
        return

    def seed(self, seed=None):
        """
        Used to set the seed of the environment,
        to reproduce the same randomness.

        :param seed: (int) specifies the seed number

        :return: (int in list) the numpy seed that you can use further.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Resets the environment to the current starting state of the environment.

        :return: (nd.array) specifies the observations returned after resetting
                            the environment. Again, it follows the
                            observation_mode specified.
        """
        self._tracker.add_episode_experience(self._episode_length)
        self._episode_length = 0
        success_signal, interventions_info, reset_observation_space_signal = \
            self._task.reset_task()
        if reset_observation_space_signal:
            self._reset_observations_space()
        # TODO: make sure that stage observations returned are up to date
        if self._data_recorder:
            self._data_recorder.new_episode(
                self.get_current_state_variables(),
                task_name=self._task._task_name,
                task_params=self._task.get_task_params(),
                world_params=self.get_world_params())
        if self._observation_mode == "pixel":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            return np.concatenate((current_images, goal_images), axis=0)
        else:
            return self._task.filter_structured_observations()

    def set_starting_state(self, interventions_dict=None, check_bounds=True):
        """
        Used to intervene on the starting state of the environment.

        :param interventions_dict: (dict) specifies the state variables that
                                          you want to intervene on as well as
                                          values of the interventions.
        :param check_bounds: (bool) specified when not in train mode and a
                                    check for the intervention if its allowed
                                    or not is needed.

        :return: (bool) indicating if an intervention was successful or not.
        """
        interventions_dict = copy.deepcopy(interventions_dict)
        self._tracker.do_intervention(self._task, interventions_dict)
        success_signal, interventions_info, reset_observation_space_signal = \
            self._task.reset_task(interventions_dict,
                                  check_bounds=check_bounds)
        if reset_observation_space_signal:
            self._reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                logging.warning("Invalid Intervention was just executed!")
                self._tracker.add_invalid_intervention(interventions_info)
        obs = self.reset()
        return success_signal, obs

    def close(self):
        """
        closes the environment in a safe manner should be called at the
        end of the program.

        :return: None
        """
        if self._data_recorder:
            self._data_recorder.save()
        self._robot.close()

    def _get_tracker(self):
        """

        :return: (causal_world.loggers.Tracker) returns a tracker that logs a
                                                lot of the stuff that happens
                                                during the usage of the
                                                environment.
        """
        return self._tracker

    def _is_done(self):
        """

        :return: (bool) returns if the task is finished or not.
        """
        if self._episode_length > self._max_episode_length:
            return True
        else:
            return self._task.is_done()

    def do_single_random_intervention(self):
        """
        Performs a single random intervention on one of the available variables
        to intervene on, which are the variables included in space A or space B.
        Depending if the env is using train_space_only or not.

        :return: (dict) The intervention that was performed.
        :return: (bool) indicating if an intervention was successful or not.
        :return: (nd.array) the observations after performing the intervention.
        """
        success_signal, interventions_info, interventions_dict, reset_observation_space_signal = \
            self._task.do_single_random_intervention()
        if reset_observation_space_signal:
            self._reset_observations_space()
        if len(interventions_dict) > 0:
            self._tracker.do_intervention(self._task, interventions_dict)
            if success_signal is not None:
                if not success_signal:
                    logging.warning("Invalid Intervention was just executed!")
                    self._tracker.add_invalid_intervention(interventions_info)
        if self._observation_mode == "pixel":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            obs = np.concatenate((current_images, goal_images), axis=0)
        else:
            obs = self._task.filter_structured_observations()
        return interventions_dict, success_signal, obs

    def do_intervention(self, interventions_dict, check_bounds=True):
        """
        Performs interventions on variables specified by the intervention dict
        to intervene on.

        :param interventions_dict: (dict) The variables to intervene on and the
                                          values of the intervention.
        :param check_bounds: (bool) specified when not in train mode and a
                                    check for the intervention if its allowed
                                    or not is needed.

        :return: (bool) indicating if an intervention was successful or not.
        :return: (nd.array) the observations after performing the intervention.
        """
        success_signal, interventions_info, reset_observation_space_signal = \
            self._task.apply_interventions(copy.deepcopy(interventions_dict),
                                           check_bounds=check_bounds)
        self._tracker.do_intervention(self._task, interventions_dict)
        if reset_observation_space_signal:
            self._reset_observations_space()
        if success_signal is not None:
            if not success_signal:
                logging.warning("Invalid Intervention was just executed!")
                self._tracker.add_invalid_intervention(interventions_info)
        if self._observation_mode == "pixel":
            current_images = self._robot.get_current_camera_observations()
            goal_images = self._stage.get_current_goal_image()
            obs = np.concatenate((current_images, goal_images), axis=0)
        else:
            obs = self._task.filter_structured_observations()
        return success_signal, obs

    def get_state(self):
        """
        Returns the current state of the environment, can be used to save the
        state.

        Note: Setting state and getting state doesnt work when there is an
        intermediate intervention

        :return: (dict) specifying the state elements as a snapshot.
        """
        state = dict()
        state['pybullet_state'] = self._task._save_pybullet_state()
        state['control_index'] = self._robot._control_index
        return state
        # return self._task.save_state()

    def set_state(self, new_full_state):
        """
        Restores the state of the environment which is passed as an argument.

        Note: Setting state and getting state doesnt work when there is an
        intermediate intervention

        :param new_full_state: (dict) specifying the state elements as a snapshot.

        :return: None
        """
        self._task._restore_pybullet_state(new_full_state['pybullet_state'])
        self._robot._control_index = new_full_state['control_index']
        self._robot.update_latest_full_state()
        return

    def render(self, mode="human"):
        """
        Returns an RGB image taken from above the platform.

        :param mode: (str) not taken in account now.

        :return: (nd.array) an RGB image taken from above the platform.
        """
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        (_, _, px, _, _) = pybullet.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client)
        rgb_array = np.array(px)
        if rgb_array.ndim == 1:
            rgb_array = rgb_array.reshape((self._render_height, self._render_width, 4))
        rgb_array = np.asarray(rgb_array, dtype='uint8')
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _setup_viewing_camera(self):
        """
        Sets up the viewing camera that is used for the render function.

        :return:
        """
        if self._pybullet_client_w_o_goal_id is not None:
            client = self._pybullet_client_w_o_goal_id
        else:
            client = self._pybullet_client_full_id
        self._cam_dist = 0.6
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
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=client)
        return

    def get_current_state_variables(self):
        """
        Returns a snapshot of the world at that point in time, includes all
        variables that can be intervened on and exposed in the environment.

        :return: (dict) specifying all state variables along with their values.
        """
        return dict(self._task.get_current_variable_values())

    def get_world_params(self):
        """
        Returns the current params of the world that were supposed to be passed
        to the CausalWorld object or manipulated in a further function.

        :return: (dict) describing the params that can be used to recreate the
                        CausalWorld object not including the task.
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
        world_params["wrappers"] = self._wrappers_dict
        return world_params

    def add_wrapper_info(self, wrapper_dict):
        """
        Adds wrapper info to the wrappers property such that the world can
        be saved with the wrappers and loaded afterwards.

        :param wrapper_dict: (dict) a dict specifying the wrapper info.

        :return: None
        """
        self._wrappers_dict.update(wrapper_dict)
        return

    def save_world(self, log_relative_path):
        """
        saves the tracker object of the world that tracks different aspects of
        the current running world.

        :param log_relative_path: (str) relative path to save the tracker
                                           object in

        :return: None
        """
        if not os.path.exists(log_relative_path):
            os.makedirs(log_relative_path)
        tracker_path = os.path.join(log_relative_path, 'tracker')
        tracker = self._get_tracker()
        tracker.save(file_path=tracker_path)
        return

    def get_variable_space_used(self):
        """
        :return: (str) specifying the space that is currently used by the
                       environment.
        """
        return self._task.get_variable_space_used()

    def set_intervention_space(self, variables_space):
        """

        :param variables_space: (str) "space_a", "space_b" or "space_a_b"

        :return:
        """
        self._task.set_intervention_space(variables_space)
        return

    def get_intervention_space_a(self):
        """
        :return: (dict) specifies the variables and their corresponding bounds
                        in space A.
        """
        return self._task.get_intervention_space_a()

    def get_intervention_space_b(self):
        """
        :return: (dict) specifies the variables and their corresponding bounds
                        in space B.
        """
        return self._task.get_intervention_space_b()

    def get_intervention_space_a_b(self):
        """

        :return: (dict) specifies the variables and their corresponding bounds
                        in space A_B.
        """
        return self._task.get_intervention_space_a_b()

    def get_joint_positions_raised(self):
        """
        :return: (nd.array) specifying the 9 joint positions of the
                            raised fingers.
        """
        return self._robot._robot_actions.\
            joint_positions_raised

    def get_action_mode(self):
        """
        :return: (str) returns which action mode the causal_world is operating in.
        """
        return self._action_mode

    def set_action_mode(self, action_mode):
        """
        used to set or change the action mode.

        :param action_mode: (str) action mode to operate in.

        :return: None
        """
        self._action_mode = action_mode
        self._robot.set_action_mode(action_mode)
        return

    def get_robot(self):
        """

        :return: (causal_world.TriFingerRobot) returns the robot object of
                                                  the causal_world.
        """
        return self._robot

    def get_task(self):
        """

        :return: (causal_world.BaseTask) returns the task object of
                                            the causal_world.
        """
        return self._task

    def get_stage(self):
        """

        :return: (causal_world.Stage) returns the stage object of
                                         the causal_world.
        """
        return self._stage

    def get_tracker(self):
        """

        :return: (causal_world.Tracker) returns the tracker
                                           object of the causal_world.
        """
        return self._tracker

    def _create_world(self, initialize_goal_image=False):
        """
        This function loads the urdfs of the robot in all the pybullet clients

        :param initialize_goal_image: (bool) used to indicate if pybullet client
                                             repsonsible for the goal image needs
                                             to be initialized.
        :return:
        """
        self._reset_world()
        finger_base_position = [0, 0, 0.0]
        finger_base_orientation = [0, 0, 0, 1]
        if initialize_goal_image:
            client_list = [
                self._pybullet_client_w_o_goal_id,
                self._pybullet_client_w_goal_id, self._pybullet_client_full_id
            ]
        else:
            client_list = [
                self._pybullet_client_w_o_goal_id, self._pybullet_client_full_id
            ]

        for client in client_list:
            if client is not None:
                pybullet.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                                 physicsClientId=client)
                pybullet.setGravity(0, 0, -9.81, physicsClientId=client)
                pybullet.setTimeStep(self._simulation_time,
                                     physicsClientId=client)
                pybullet.loadURDF("plane_transparent.urdf", [0, 0, 0],
                                  physicsClientId=client)
                pybullet.loadURDF(fileName=self._finger_urdf_path,
                                  basePosition=finger_base_position,
                                  baseOrientation=finger_base_orientation,
                                  useFixedBase=1,
                                  flags=(pybullet.URDF_USE_INERTIA_FROM_FILE |
                                         pybullet.URDF_USE_SELF_COLLISION),
                                  physicsClientId=client)
                if self.link_name_to_index is None:
                    self.link_name_to_index = {
                        pybullet.getBodyInfo(WorldConstants.ROBOT_ID,
                                             physicsClientId=client)[0].decode("UTF-8"):
                            -1,
                    }
                    for joint_idx in range(
                            pybullet.getNumJoints(WorldConstants.ROBOT_ID,
                                                  physicsClientId=client)):
                        link_name = pybullet.getJointInfo(
                            WorldConstants.ROBOT_ID,
                            joint_idx,
                            physicsClientId=client)[12].decode("UTF-8")
                        self.link_name_to_index[link_name] = joint_idx

                    self._revolute_joint_ids = [
                        self.link_name_to_index[name]
                        for name in WorldConstants.JOINT_NAMES
                    ]
                    self.finger_tip_ids = [
                        self.link_name_to_index[name]
                        for name in WorldConstants.TIP_LINK_NAMES
                    ]
                    self.finger_link_ids = self._revolute_joint_ids
                    self.last_joint_position = [0] * len(
                        self._revolute_joint_ids)
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
                        physicsClientId=client)
                self._create_stage(client)
        return

    def _reset_world(self):
        """
        resets the world itself by resetting the simulation too.

        :return:
        """
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
        """
        creates the stage of the simulation by loading the .stl files.

        :param pybullet_client: (int) the pybuller client to create the
                                      stage in.
        :return:
        """

        def mesh_path(filename):
            return os.path.join(self._robot_properties_path, "meshes", "stl",
                                filename)

        table_colour = (0.31, 0.27, 0.25, 1.0)
        high_border_colour = (0.95, 0.95, 0.95, 1.0)
        floor_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=mesh_path("trifinger_table_without_border.stl"),
            flags=0,
            physicsClientId=pybullet_client)
        obj = pybullet.createMultiBody(baseCollisionShapeIndex=floor_id,
                                       baseVisualShapeIndex=-1,
                                       basePosition=[0, 0, 0.01],
                                       baseOrientation=[0, 0, 0, 1],
                                       physicsClientId=pybullet_client)
        pybullet.changeVisualShape(obj,
                                   -1,
                                   rgbaColor=table_colour,
                                   physicsClientId=pybullet_client)

        stage_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=mesh_path("edu/frame_wall.stl"),
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=pybullet_client)
        obj = pybullet.createMultiBody(baseCollisionShapeIndex=stage_id,
                                       baseVisualShapeIndex=-1,
                                       basePosition=[0, 0, 0.01],
                                       baseOrientation=[0, 0, 0, 1],
                                       physicsClientId=pybullet_client)
        pybullet.changeVisualShape(obj,
                                   -1,
                                   rgbaColor=high_border_colour,
                                   physicsClientId=pybullet_client)
        return

    def _instantiate_pybullet(self):
        """
        This function is used for instantiating all pybullet instances used for
        the current simulation

        :return:
        """
        if self._observation_mode == 'pixel':
            self._pybullet_client_w_o_goal_id = pybullet.connect(
                pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_w_o_goal_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_o_goal_id)
            self._pybullet_client_w_goal_id = pybullet.connect(pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_w_goal_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_w_goal_id)
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(pybullet.GUI)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_GUI,
                    0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                    0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                    0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                    0,
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.resetSimulation(
                    physicsClientId=self._pybullet_client_full_id)
                pybullet.setPhysicsEngineParameter(
                    deterministicOverlappingPairs=1,
                    physicsClientId=self._pybullet_client_full_id)
        else:
            if self._enable_visualization:
                self._pybullet_client_full_id = pybullet.connect(pybullet.GUI)
            else:
                self._pybullet_client_full_id = pybullet.connect(
                    pybullet.DIRECT)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI,
                0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,
                0,
                physicsClientId=self._pybullet_client_full_id)
            pybullet.resetSimulation(
                physicsClientId=self._pybullet_client_full_id)
            pybullet.setPhysicsEngineParameter(
                deterministicOverlappingPairs=1,
                physicsClientId=self._pybullet_client_full_id)
        return
