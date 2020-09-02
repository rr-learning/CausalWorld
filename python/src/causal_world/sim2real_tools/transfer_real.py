import robot_fingers
import robot_interfaces
import numpy as np


class TransferReal(object):
    def __init__(self, env):
        """

        :param env:
        """
        self._env = env
        self._real_robot = robot_fingers.Robot(robot_interfaces.trifinger,
                                              robot_fingers.create_trifinger_backend,
                                              "trifinger.yml")
        self._real_robot.initialize()
        self._frontend = self._real_robot.frontend
        self._repetitions = 1000.0 / (self._env._simulation_time /
                                      self._env._skip_frame)

    def step(self, action):
        """

        :param action:
        :return:
        """
        for i in range(self._repetitions):
            t = self._frontend.append_desired_action(
                robot_interfaces.trifinger.Action(position=action))
            self._frontend.wait_until_time_index(t)
        current_position = self._frontend.get_observation(t).position
        current_velocity = self._frontend.get_observation(t).velocity
        current_torque = self._frontend.get_observation(t).torque
        obs = np.array([current_position, current_velocity, current_torque])
        return obs.flatten()

    def reset(self):
        raise Exception("Not implemented yet")
