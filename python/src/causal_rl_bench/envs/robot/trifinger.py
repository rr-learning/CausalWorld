from causal_rl_bench.envs.robot.action import TriFingerAction
from causal_rl_bench.envs.robot.observations import TriFingerObservations


class TriFingerRobot(object):
    def __init__(self, action_mode, observation_mode,
                 normalize_actions=True, control_rate=0.003):
        self.robot_actions = TriFingerAction(action_mode)
        self.robot_observations = TriFingerObservations(observation_mode)
        pass

    def set_action_mode(self):
        raise Exception(" Not implemented")

    def get_action_mode(self):
        raise Exception(" Not implemented")

    def set_observation_mode(self):
        raise Exception(" Not implemented")

    def get_observation_mode(self):
        raise Exception(" Not implemented")

    def set_camera_rate(self):
        raise Exception(" Not implemented")

    def get_camera_rate(self):
        raise Exception(" Not implemented")

    def set_control_rate(self):
        raise Exception(" Not implemented")

    def get_control_rate(self):
        raise Exception(" Not implemented")

    def turn_on_cameras(self):
        raise Exception(" Not implemented")

    def turn_off_cameras(self):
        raise Exception(" Not implemented")

    def apply_action(self):
        raise Exception(" Not implemented")

    def get_full_state(self):
        raise Exception(" Not implemented")

    def set_full_state(self):
        raise Exception(" Not implemented")

    def reset_robot_state(self):
        raise Exception(" Not implemented")

    def get_last_action_applied(self):
        raise Exception(" Not implemented")

    def get_current_full_observations(self):
        raise Exception(" Not implemented")

    def get_current_partial_observations(self, keys):
        raise Exception(" Not implemented")