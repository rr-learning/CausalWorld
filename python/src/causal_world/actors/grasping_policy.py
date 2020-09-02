import numpy as np


class GraspingPolicy(object):
    def __init__(self, tool_blocks_order):
        """

        :param tool_blocks_order:
        """
        self._program_counter = 0
        self._program = tool_blocks_order
        self._phase = 0
        self._t = 0

        self._h0_r = -0.98
        self._h1_r = -0.4
        self._h0_gb = -0.98
        self._h1_gb = -0.4

        self._d0_r = 0.038
        self._d0_gb = 0.038
        self._d1_r = 0.1
        self._d1_gb = 0.1

        self._a1 = np.pi / 2
        self._gb_angle_spread = 0.8 * np.pi
        self._a2 = 3 * np.pi / 2 + (self._gb_angle_spread / 2)
        self._a3 = 3 * np.pi / 2 - (self._gb_angle_spread / 2)

        self._fall_trigger_h = -0.7
        self._phase_velocities = [0.008, 0.01, 0.02,
                                  0.005, 0.005, 0.01,
                                  0.01]

        pass

    def act(self, obs):
        """
        The function is called for the agent to act in the world.

        :param obs: (nd.array) defines the observations received by the agent
                               at time step t

        :return: (nd.array) defines the action to be executed at time step t
        """
        if self._program_counter == len(self._program):
            return obs[19:28]

        block_idx = self._program[self._program_counter]
        number_of_blocks = 2

        target_height = obs[28 + (number_of_blocks*17) + (block_idx*11) + 6]

        target_x = obs[28 + (number_of_blocks*17) + (block_idx*11) + 4]
        target_y = obs[28 + (number_of_blocks*17) + (block_idx*11) + 5]

        current_cube_x = obs[28 + (block_idx*17) + 4]
        current_cube_y = obs[28 + (block_idx*17) + 5]

        next_cube_x = 0
        next_cube_y = 0

        # Detect falling cube
        if self._phase == 3 and obs[28 + (block_idx*17) + 6] < self._fall_trigger_h:
            self._phase = 0
            self._t = 0

        self._t += self._phase_velocities[self._phase]
        if self._t >= 1.0:
            self._phase += 1
            self._t -= 1.0

        if self._phase >= 7:
            self._phase = 0
            self._program_counter += 1
            self._t = 0
            self.act(obs)

        #calculate the target of the grip center
        # xy_target is the target of the grip center
        current_xy_target = self._get_current_xy_target(target_x,
                                                        target_y,
                                                        current_cube_x,
                                                        current_cube_y,
                                                        next_cube_x,
                                                        next_cube_y)

        #calculate end_effector_positions
        # Target heights of fingertips
        target_h_r, target_h_g, target_h_b = self._get_target_hs(target_height)

        # Target-distance of fingertips from the grip center
        d_r, d_g, d_b = self._get_ds()
        #construct end effector positions target
        # Construct full target positions for each fingertip
        pos_r = np.array([current_xy_target[0] + d_r * np.cos(self._a1),
                          current_xy_target[1] + d_r * np.sin(self._a1),
                          target_h_r])
        pos_g = np.array([current_xy_target[0] + d_g * np.cos(self._a2),
                          current_xy_target[1] + d_g * np.sin(self._a2),
                          target_h_g])
        pos_b = np.array([current_xy_target[0] + d_b * np.cos(self._a3),
                          current_xy_target[1] + d_b * np.sin(self._a3),
                          target_h_b])
        return np.concatenate((pos_r, pos_g, pos_b), axis=0)

    def _get_ds(self):
        """

        :return:
        """
        if self._phase == 0:
            d_r = self._d1_r
            d_gb = self._d1_gb
        elif self._phase == 1:
            a = self._mix_sin(max(0, 2 * (self._t - 0.5)))
            d_r = self._combine_convex(self._d1_r, self._d0_r, a)
            d_gb = self._combine_convex(self._d1_gb, self._d0_gb, a)
        elif self._phase in [2, 3]:
            d_r = self._d0_r
            d_gb = self._d0_gb
        elif self._phase == 4:
            d_r = self._d0_r
            d_gb = self._d0_gb
        elif self._phase in [5, 6]:
            d_r = self._d1_r
            d_gb = self._d1_gb
        else:
            raise ValueError()
        return [d_r, d_gb, d_gb]

    def _get_current_xy_target(self, target_x,
                               target_y, current_cube_x,
                               current_cube_y,
                               next_cube_x,
                               next_cube_y):
        """

        :param target_x:
        :param target_y:
        :param current_cube_x:
        :param current_cube_y:
        :param next_cube_x:
        :param next_cube_y:
        :return:
        """
        if self._phase < 4:
            current_x = current_cube_x
            current_y = current_cube_y
        else:
            current_x = next_cube_x
            current_y = next_cube_y

        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + \
                    alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        """

        :return:
        """
        if self._phase < 3:
            return 0
        elif self._phase == 3:
            return self._mix_sin(self._t)
        elif self._phase == 4:
            return 1.0
        elif self._phase == 5:
            return 1.0
        elif self._phase == 6:
            return 1 - self._mix_sin(self._t)
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        """

        :param target_height:
        :return:
        """
        if self._phase == 0:
            h_r = self._h1_r
            h_gb = self._h1_gb
        elif self._phase == 1:
            a = self._mix_sin(max(0, self._t))
            h_r = self._combine_convex(self._h1_r, self._h0_r, a)
            h_gb = self._combine_convex(self._h1_gb, self._h0_gb, a)
        elif self._phase == 2:
            a = self._mix_sin(max(0, self._t))
            h_r = self._combine_convex(self._h0_r, self._h1_r, a)
            h_gb = self._combine_convex(self._h0_gb, self._h1_gb, a)
        elif self._phase == 3:
            h_r = self._h1_r
            h_gb = self._h1_gb
        elif self._phase == 4:
            h_target_r = target_height
            h_target_gb = h_target_r + (self._h0_gb - self._h0_r)
            h_r = self._combine_convex(self._h1_r, h_target_gb,
                                       self._mix_sin(self._t))
            h_gb = self._combine_convex(self._h1_gb, h_target_gb,
                                        self._mix_sin(self._t))
        elif self._phase == 5:
            h_target_r = target_height
            h_target_gb = h_target_r + (self._h0_gb - self._h0_r)
            h_r = self._combine_convex(h_target_r, self._h1_r,
                                       self._mix_sin(self._t))
            h_gb = self._combine_convex(h_target_gb, self._h1_gb,
                                        self._mix_sin(self._t))
        elif self._phase == 6:
            h_r = self._h1_r
            h_gb = self._h1_gb
        else:
            raise ValueError()

        return np.array([h_r, h_gb, h_gb])

    def reset_controller(self):
        """

        :return:
        """
        self._phase = 0
        self._t = 0
        self._program_counter = 0

    def _mix_sin(self, t):
        """

        :param t:
        :return:
        """
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        """

        :param a:
        :param b:
        :param alpha:
        :return:
        """
        return (1 - alpha) * a + alpha * b

