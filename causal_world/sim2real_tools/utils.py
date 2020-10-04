import gym


class RealisticRobotWrapper(gym.Wrapper):

    def __init__(self, env):
        """
        This wrapper makes the simulated environment close to the real robot.

        :param env: (causal_world.CausalWorld) the environment to make realistic.
        """
        # TODO: this wrapper can't be loaded at the moment or saved
        super(RealisticRobotWrapper, self).__init__(env)
        self.env.set_starting_state({
            # 'robot_height': 0.286,
            'robot_finger_60_link_0': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_60_link_1': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_60_link_2': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_60_link_3': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_120_link_0': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_120_link_1': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_120_link_2': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_120_link_3': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_300_link_0': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_300_link_1': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_300_link_2': {
                'color': [0.2, 0.2, 0.2]
            },
            'robot_finger_300_link_3': {
                'color': [0.2, 0.2, 0.2]
            }
        })
        return
