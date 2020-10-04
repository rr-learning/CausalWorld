import numpy as np
from causal_world.actors.base_policy import BaseActorPolicy


class GraspingPolicy(BaseActorPolicy):
    """
    This policy is expected to run @25 Hz, its a hand designed policy for
    picking and placing blocks of a specific size 6.5CM weighing 20grams
    for the best result tried.
    The policy outputs desired normalized end_effector_positions

    Description of phases:
    - Phase 0: Move finger-center above the cube center of the current
               instruction.
    - Phase 1: Lower finger-center down to encircle the target cube, and
               close grip.
    - Phase 2: Move finger-center up again, keeping the grip tight
              (lifting the block).
    - Phase 3: Smoothly move the finger-center toward the goal xy, keeping the
               height constant.
    - Phase 4: Move finger-center vertically toward goal height
                (keeping relative difference of different finger heights given
                by h0), at the same time loosen the grip (i.e. increasing the
                radius of the "grip circle").
    - Phase 5: Move finger center up again

    Other variables and values:
    - alpha: interpolation value between two positions
    - ds: Distances of finger tips to grip center
    - t: time between 0 and 1 in current phase
    - phase: every instruction has 7 phases (described above)
    - program_counter: The index of the current instruction in the overall
                       program. Is incremented once the policy has successfully
                       completed all phases.

    Hyperparameters:

    - phase_velocity_k : the speed at which phase "k" in the state machine
                         progresses.
    - d0_r, d0_gb: Distance of finger tips from grip center while gripping the
                   object.
    - gb_angle_spread: Angle between green and blue finger tips along the "grip
                       circle".
    - d1_r, d1_gb: Distance of finger tips from grip center while not gripping
    - h1_r, h1_gb: Height of grip center while moving around
    - h0_r, h0_gb: Height of grip center to which it is lowered while grasping
    - fall_trigger_h: if box is detected below this height when it is supposed
                      to be gripped, try grasping it again (reset phase to 0).

    """
    def __init__(self, tool_blocks_order):
        """

        :param tool_blocks_order: (nd.array) specifies the program where the
                                             indicies ranges from 0 to the
                                             number of blocks available in the
                                             arena.
        """
        super(GraspingPolicy, self).__init__(identifier="grasping_policy")
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

        self.current_target_x = None
        self.current_target_y = None

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
        number_of_blocks = len(self._program)

        target_height = obs[28 + (number_of_blocks*17) + (block_idx*11) + 6]

        target_x = obs[28 + (number_of_blocks*17) + (block_idx*11) + 4]
        target_y = obs[28 + (number_of_blocks*17) + (block_idx*11) + 5]

        # Only set target for cube when phase is 0, otherwise the robot moving
        # the target cube creates a runaway effect due to a moving target
        if self._phase == 0:
            self.current_target_x = obs[28 + (block_idx * 17) + 4]
            self.current_target_y = obs[28 + (block_idx * 17) + 5]

        if self._program_counter < len(self._program) - 1:
            next_block_idx = self._program[self._program_counter + 1]
        else:
            next_block_idx = self._program[-1]

        next_cube_x = obs[28 + (next_block_idx*17) + 4]
        next_cube_y = obs[28 + (next_block_idx*17) + 5]

        # Detect falling cube
        if self._phase == 3 and obs[28 + (block_idx*17) + 6] < self._fall_trigger_h:
            self._phase = 0
            self._t = 0

        #calculate the target of the grip center
        interpolated_xy = self._get_interpolated_xy(target_x,
                                                    target_y,
                                                    self.current_target_x,
                                                    self.current_target_y,
                                                    next_cube_x,
                                                    next_cube_y)

        # Target heights of fingertips
        target_h_r, target_h_g, target_h_b = self._get_target_hs(target_height)

        # Target-distance of fingertips from the grip center
        d_r, d_g, d_b = self._get_ds()
        # Construct full target positions for each fingertip
        pos_r = np.array([interpolated_xy[0] + d_r * np.cos(self._a1),
                          interpolated_xy[1] + d_r * np.sin(self._a1),
                          target_h_r])
        pos_g = np.array([interpolated_xy[0] + d_g * np.cos(self._a2),
                          interpolated_xy[1] + d_g * np.sin(self._a2),
                          target_h_g])
        pos_b = np.array([interpolated_xy[0] + d_b * np.cos(self._a3),
                          interpolated_xy[1] + d_b * np.sin(self._a3),
                          target_h_b])

        self._t += self._phase_velocities[self._phase]
        if self._t >= 1.0:
            self._phase += 1
            self._t -= 1.0

        if self._phase >= 7:
            self._phase = 0
            self._program_counter += 1
            self._t = 0

        return np.concatenate((pos_r, pos_g, pos_b), axis=0)

    def _get_ds(self):
        """
        :return: distances of finger tips to grip center
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

    def _get_interpolated_xy(self, target_x,
                             target_y, current_cube_x,
                             current_cube_y,
                             next_cube_x,
                             next_cube_y):
        """
        :param target_x: target x of the grip center.
        :param target_y: target y of the grip center.
        :param current_cube_x: x of current cube to be gripped.
        :param current_cube_y: y of current cube to be gripped.
        :param next_cube_x: x of next cube to be gripped.
        :param next_cube_y: y of next cube to be gripped.
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
        :return: alpha for interpolation depending on the phase.
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
        :param target_height: target height to be reached.
        :return: target height for all the end effectors.
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

    def reset(self):
        """
        resets the controller

        :return:
        """
        self._phase = 0
        self._t = 0
        self._program_counter = 0

    def _mix_sin(self, t):
        """
        :param t: time ranging from 0 to 1.
        :return: mixed sin wave.
        """
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        """
        :param a: start
        :param b: end
        :param alpha: interpolation
        :return: convex combination.
        """
        return (1 - alpha) * a + alpha * b

