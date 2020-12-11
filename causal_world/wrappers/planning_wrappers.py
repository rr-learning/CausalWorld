from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
from causal_world.utils.rotation_utils import quaternion_to_euler, \
    euler_to_quaternion
import gym
import numpy as np
from causal_world.configs.world_constants import WorldConstants
from gym import spaces


class ObjectSelectorActorPolicy(BaseInterventionActorPolicy):

    def __init__(self):
        """

        """
        super(ObjectSelectorActorPolicy, self).__init__()
        self.low_joint_positions = None
        self.current_action = None
        self.selected_object = None

    def initialize_actor(self, env):
        """

        :param env:
        :return:
        """
        self.low_joint_positions = env.get_joint_positions_raised()
        return

    def add_action(self, action, selected_object):
        """

        :param action:
        :param selected_object:
        :return:
        """
        self.current_action = action
        self.selected_object = selected_object

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        # action 1 is [0, 1, 2, 3, 4, 5, 6] 0 do nothing, 1 go
        #  up 2 down, 3 right, 4 left, 5, front, 6 back
        # action 2 is [0, 1, 2, 3, 4, 5, 6] 0 do nothing, 1 yaw clockwise,
        # 2 yaw anticlockwise, 3 roll clockwise,
        # 4 roll anticlockwise, 5 pitch clockwise, 6 pitch anticlockwise
        interventions_dict = dict()
        # interventions_dict['joint_positions'] = self.low_joint_positions
        interventions_dict[self.selected_object] = dict()
        if self.current_action[1] != 0:
            interventions_dict[self.selected_object]['cartesian_position'] = \
                variables_dict[self.selected_object]['cartesian_position']
            if self.current_action[1] == 1:
                interventions_dict[
                    self.selected_object]['cartesian_position'][-1] += 0.002
            elif self.current_action[1] == 2:
                interventions_dict[
                    self.selected_object]['cartesian_position'][-1] -= 0.002
            elif self.current_action[1] == 3:
                interventions_dict[
                    self.selected_object]['cartesian_position'][0] += 0.002
            elif self.current_action[1] == 4:
                interventions_dict[
                    self.selected_object]['cartesian_position'][0] -= 0.002
            elif self.current_action[1] == 5:
                interventions_dict[
                    self.selected_object]['cartesian_position'][1] += 0.002
            elif self.current_action[1] == 6:
                interventions_dict[
                    self.selected_object]['cartesian_position'][1] -= 0.002
            else:
                raise Exception("The passed action mode is not supported")
        if self.current_action[2] != 0:
            euler_orientation = \
                quaternion_to_euler(variables_dict
                                    [self.selected_object]['orientation'])
            if self.current_action[2] == 1:
                euler_orientation[-1] += 0.1
            elif self.current_action[2] == 2:
                euler_orientation[-1] -= 0.1
            elif self.current_action[2] == 3:
                euler_orientation[1] += 0.1
            elif self.current_action[2] == 4:
                euler_orientation[1] -= 0.1
            elif self.current_action[2] == 5:
                euler_orientation[0] += 0.1
            elif self.current_action[2] == 6:
                euler_orientation[0] -= 0.1
            else:
                raise Exception("The passed action mode is not supported")
            interventions_dict[self.selected_object]['orientation'] = \
                euler_to_quaternion(euler_orientation)
        return interventions_dict


class ObjectSelectorWrapper(gym.Wrapper):

    def __init__(self, env):
        """

        :param env: (causal_world.CausalWorld) the environment to convert.
        """
        super(ObjectSelectorWrapper, self).__init__(env)
        self.env = env
        self.env.set_skip_frame(1)
        intervention_dict = {'robot_height': 1.0}
        env.set_starting_state(intervention_dict, check_bounds=False)
        self.intervention_actor = ObjectSelectorActorPolicy()
        self.intervention_actor.initialize_actor(self.env)
        if self.env._robot.get_observation_mode() == 'structured':
            self.observation_space = gym.spaces.Box(
                self.env.observation_space.low[28:],
                self.env.observation_space.high[28:],
                dtype=np.float64)
        self.objects_order = list(
            self.env.get_stage().get_rigid_objects().keys())
        self.objects_order.sort()
        number_of_objects = len(self.objects_order)
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(number_of_objects), gym.spaces.Discrete(7),
             gym.spaces.Discrete(7)))
        self.env.add_wrapper_info({'object_selector': dict()})

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
        self.intervention_actor.add_action(action,
                                           self.objects_order[action[0]])
        intervention_dict = self.intervention_actor.act(
            self.env.get_current_state_variables())
        self.env.do_intervention(intervention_dict, check_bounds=False)
        obs, reward, done, info = self.env.step(self.env.action_space.low)
        if self.env.get_observation_mode() == 'structured':
            obs = obs[28:]
        return obs, reward, done, info

    def reset(self):
        """
        Resets the environment to the current starting state of the environment.

        :return: (nd.array) specifies the observations returned after resetting
                            the environment. Again, it follows the
                            observation_mode specified.
        """
        result = self.env.reset()
        interventions_dict = dict()
        interventions_dict['joint_positions'] = self.intervention_actor.low_joint_positions
        self.env.do_intervention(interventions_dict, check_bounds=False)
        return result


class PickxyzDropxyzWrapper(gym.Wrapper):

    def __init__(self, env):
        """

        :param env: (causal_world.CausalWorld) the environment to convert.
        """
        super(PickxyzDropxyzWrapper, self).__init__(env)
        self.env = env
        self.env.set_skip_frame(250)
        intervention_dict = {'robot_height': 1.0}
        env.set_starting_state(intervention_dict, check_bounds=False)
        if self.env._robot.get_observation_mode() == 'structured':
            raise Exception("This wrapper is meant for pixel mode only")
        self._lower_bounds = np.append(np.tile(WorldConstants.ARENA_BB[0], 2),
                                       [-np.pi, -np.pi, -np.pi])
        self._upper_bounds = np.append(np.tile(WorldConstants.ARENA_BB[1], 2),
                                       [-np.pi, -np.pi, -np.pi])
        if self.env._normalize_actions:
            self.action_space = spaces.Box(low=-np.ones(9),
                                           high=np.ones(9),
                                           dtype=np.float64)
        else:
            self.action_space = spaces.Box(low=self._lower_bounds,
                                           high=self._upper_bounds,
                                           dtype=np.float64)
        self._number_of_cameras = self.env._camera_indicies.shape[0]
        self.env.add_wrapper_info({'PickxyzDropxyzWrapper': dict()})

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
        #either denormalize or not
        denormalized_action = np.copy(action)
        if self.env._normalize_actions:
            denormalized_action = np.copy(self._lower_bounds + (action + 1.0) / 2.0 * \
                                          (self._upper_bounds - self._lower_bounds))
        #find the object closest to the pick location and distance is smaller than this object size
        closest_object = self.env.get_stage().\
            get_tool_object_closest_to_point(denormalized_action[:3])
        if closest_object is not None:
            intervention_dict = dict()
            intervention_dict[closest_object] = \
                {'cartesian_position': denormalized_action[3:6],
                 'orientation': euler_to_quaternion(denormalized_action[6:])}
            success_signal, action_images = self.env.do_intervention(intervention_dict,
                                                                    check_bounds=False)
        else:
            action_images = self.env._robot.get_current_camera_observations()
        action_images = action_images[:self._number_of_cameras]
        obs, reward, done, info = self.env.step(np.zeros(9))
        if closest_object is not None:
            info['pick_success'] = True
        else:
            info['pick_success'] = False
        if closest_object is not None and success_signal:
            info['drop_success'] = True
        else:
            info['drop_success'] = False
        new_obs = np.concatenate([action_images,
                                  obs], axis=0)
        return new_obs, reward, done, info

    def reset(self):
        """
        Resets the environment to the current starting state of the environment.

        :return: (nd.array) specifies the observations returned after resetting
                            the environment. Again, it follows the
                            observation_mode specified.
        """
        self.env.reset()
        for i in range(2):
            obs, reward, done, info = self.env.step(self.env.action_space.low)
        return obs
