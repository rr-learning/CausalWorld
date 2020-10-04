from causal_world.task_generators.base_task import BaseTask
import numpy as np
from causal_world.utils.rotation_utils import quaternion_conjugate, \
    quaternion_mul


class PickAndPlaceTaskGenerator(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=0,
                 dense_reward_weights=np.array([750, 50, 250, 0, 0.005]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.02,
                 joint_positions=None,
                 tool_block_position=np.array([0, -0.09, 0.0325]),
                 tool_block_orientation=np.array([0, 0, 0, 1]),
                 goal_block_position=np.array([0, 0.09, 0.0325]),
                 goal_block_orientation=np.array([0, 0, 0, 1])):
        """
        This task generator generates a task of picking and placing an object
        across a fixed block in the middle of the arena.

        :param variables_space: (str) space to be used either 'space_a' or
                                      'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the volumetric
                                              fraction overlap more than 90%.
        :param tool_block_mass: (float) specifies the blocks mass.
        :param joint_positions: (nd.array) specifies the joints position to start
                                            the episode with. None if the default
                                            to be used.
        :param tool_block_position: (nd.array) specifies the cartesian position
                                               of the tool block, x, y, z.
        :param tool_block_orientation: (nd.array) specifies the euler orientation
                                               of the tool block, yaw, roll, pitch.
        :param goal_block_position: (nd.array) specifies the cartesian position
                                               of the goal block, x, y, z.
        :param goal_block_orientation: (nd.array) specifies the euler orientation
                                               of the goal block, yaw, roll, pitch.
        """
        super().__init__(task_name="pick_and_place",
                         variables_space=variables_space,
                         fractional_reward_weight=fractional_reward_weight,
                         dense_reward_weights=dense_reward_weights,
                         activate_sparse_reward=activate_sparse_reward)
        self._task_robot_observation_keys = ["time_left_for_task",
                                             "joint_positions",
                                             "joint_velocities",
                                             "end_effector_positions"]
        self._task_params["tool_block_mass"] = tool_block_mass
        self._task_params["joint_positions"] = joint_positions
        self._task_params["tool_block_position"] = tool_block_position
        self._task_params["tool_block_orientation"] = tool_block_orientation
        self._task_params["goal_block_position"] = goal_block_position
        self._task_params["goal_block_orientation"] = goal_block_orientation
        self.previous_object_position = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {
            'name': "obstacle",
            'shape': "static_cube",
            'position': [0, 0, 0.0325],
            'color': np.array([0, 0, 0]),
            'size': np.array([0.5, 0.015, 0.065])
        }
        self._stage.add_rigid_general_object(**creation_dict)
        creation_dict = {
            'name': "tool_block",
            'shape': "cube",
            'initial_position': self._task_params["tool_block_position"],
            'initial_orientation': self._task_params["tool_block_orientation"],
            'mass': self._task_params["tool_block_mass"]
        }
        self._stage.add_rigid_general_object(**creation_dict)
        creation_dict = {
            'name': "goal_block",
            'shape': "cube",
            'position': self._task_params["goal_block_position"],
            'orientation': self._task_params["goal_block_orientation"]
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        self._task_stage_observation_keys = [
            "tool_block_type", "tool_block_size",
            "tool_block_cartesian_position", "tool_block_orientation",
            "tool_block_linear_velocity", "tool_block_angular_velocity",
            "goal_block_type", "goal_block_size",
            "goal_block_cartesian_position", "goal_block_orientation",
            "obstacle_type", "obstacle_size", "obstacle_cartesian_position",
            "obstacle_orientation"
        ]
        return

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(PickAndPlaceTaskGenerator, self)._set_intervention_space_a()
        self._intervention_space_a['obstacle']['size'] = \
            np.array([self._stage.get_object_state('obstacle', 'size'),
                      self._stage.get_object_state('obstacle', 'size')])
        self._intervention_space_a['obstacle']['size'][0][-1] = 0.02
        self._intervention_space_a['obstacle']['size'][1][-1] = 0.065
        self._intervention_space_a['tool_block']['cylindrical_position'][0] = \
            np.array([0.07, np.pi/6, self._stage.get_object_state('tool_block', 'size')[-1] / 2.0])

        self._intervention_space_a['tool_block']['cylindrical_position'][1] = \
            np.array([0.12, (5 / 6.0) * np.pi ,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])
        self._intervention_space_a['goal_block']['cylindrical_position'][0] = \
            np.array([0.07, -np.pi / 6,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])

        self._intervention_space_a['goal_block']['cylindrical_position'][1] = \
            np.array([0.12, (5 / 6.0) * -np.pi,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(PickAndPlaceTaskGenerator, self)._set_intervention_space_b()
        self._intervention_space_b['obstacle']['size'] = \
            np.array([self._stage.get_object_state('obstacle', 'size'),
                      self._stage.get_object_state('obstacle', 'size')])
        self._intervention_space_b['obstacle']['size'][0][-1] = 0.065
        self._intervention_space_b['obstacle']['size'][1][-1] = 0.1
        self._intervention_space_b['tool_block']['cylindrical_position'][0] = \
            np.array([0.12, np.pi / 6,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])

        self._intervention_space_b['tool_block']['cylindrical_position'][1] = \
            np.array([0.15, (5 / 6.0) * np.pi,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])
        self._intervention_space_b['goal_block']['cylindrical_position'][0] = \
            np.array([0.12, -np.pi / 6,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])

        self._intervention_space_b['goal_block']['cylindrical_position'][1] = \
            np.array([0.15, (5 / 6.0) * -np.pi,
                      self._stage.get_object_state('tool_block', 'size')[
                          -1] / 2.0])
        return

    def _set_task_state(self):
        """

        :return:
        """
        self.previous_end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = \
            self._stage.get_object_state('tool_block', 'cartesian_position')
        self.previous_object_orientation = \
            self._stage.get_object_state('tool_block', 'orientation')
        self.previous_joint_velocities = \
            self._robot.get_latest_full_state()['velocities']
        return

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return "Task where the goal is to pick a " \
               "cube and then place it in the other side of the wall"

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        # rewards order
        # 1) delta how much the fingers are close to block
        # 2) delta how much are you getting the block close to the goal
        # 3) delta how much are you getting the block close to the target_height
        # 4) delta in orientations
        # 5) delta in joint velocities
        joint_velocities = self._robot.get_latest_full_state()['velocities']
        rewards = list()
        block_position = self._stage.get_object_state('tool_block',
                                                      'cartesian_position')
        block_orientation = self._stage.get_object_state(
            'tool_block', 'orientation')
        goal_position = self._stage.get_object_state('goal_block',
                                                     'cartesian_position')
        goal_orientation = self._stage.get_object_state('goal_block',
                                                        'orientation')
        end_effector_positions = self._robot.get_latest_full_state(
        )['end_effector_positions']
        end_effector_positions = end_effector_positions.reshape(-1, 3)

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     block_position)
        previous_distance_from_block = np.linalg.norm(
            self.previous_end_effector_positions -
            self.previous_object_position)
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)

        # calculate second reward term
        previous_dist_to_goal = np.linalg.norm(
            goal_position[:2] - self.previous_object_position[:2])
        current_dist_to_goal = np.linalg.norm(goal_position[:2] -
                                              block_position[:2])
        rewards.append(previous_dist_to_goal - current_dist_to_goal)

        # calculate third reward term (height trajectory)
        if np.sign(block_position[1]) != np.sign(goal_position[1]):
            target_height = 0.15
        else:
            target_height = self._stage.get_object_state('goal_block', 'size')[-1] / 2.0
        previous_block_to_goal = abs(self.previous_object_position[2] -
                                     target_height)
        current_block_to_goal = abs(block_position[2] - target_height)

        height_reward = previous_block_to_goal - current_block_to_goal

        rewards.append(height_reward)

        # calculate forth reward term
        quat_diff_old = quaternion_mul(
            np.expand_dims(goal_orientation, 0),
            quaternion_conjugate(
                np.expand_dims(self.previous_object_orientation, 0)))
        angle_diff_old = 2 * np.arccos(np.clip(quat_diff_old[:, 3], -1., 1.))

        quat_diff = quaternion_mul(
            np.expand_dims(goal_orientation, 0),
            quaternion_conjugate(np.expand_dims(block_orientation, 0)))
        current_angle_diff = 2 * np.arccos(np.clip(quat_diff[:, 3], -1., 1.))

        rewards.append(angle_diff_old[0] - current_angle_diff[0])
        # calculate fifth reward term
        rewards.append(-np.linalg.norm(joint_velocities -
                                       self.previous_joint_velocities))
        update_task_info = {
            'current_end_effector_positions': end_effector_positions,
            'current_tool_block_position': block_position,
            'current_tool_block_orientation': block_orientation
        }
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        """

        :param update_task_info:
        :return:
        """
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_object_position = \
            update_task_info['current_tool_block_position']
        self.previous_object_orientation = \
            update_task_info['current_tool_block_orientation']
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        """

        :param interventions_dict:

        :return:
        """
        if 'goal_block' in interventions_dict:
            if 'size' in interventions_dict['goal_block']:
                if 'tool_block' not in interventions_dict:
                    interventions_dict['tool_block'] = dict()
                interventions_dict['tool_block']['size'] = \
                    interventions_dict['goal_block']['size']
                if 'cartesian_position' not in interventions_dict['tool_block'] and \
                        'cylindrical_position' not in interventions_dict['tool_block']:
                    cyl_pos_tool = self._stage.get_object_state('tool_block', 'cylindrical_position')
                    cyl_pos_tool[-1] = interventions_dict['goal_block']['size'][-1] / 2.0
                    interventions_dict['tool_block']['cylindrical_position'] = cyl_pos_tool
                if 'cartesian_position' not in interventions_dict['goal_block'] and \
                        'cylindrical_position' not in interventions_dict['goal_block']:
                    cyl_pos_goal = self._stage.get_object_state('goal_block', 'cylindrical_position')
                    cyl_pos_goal[-1] = interventions_dict['goal_block']['size'][-1] / 2.0
                    interventions_dict['goal_block']['cylindrical_position'] = cyl_pos_goal
        elif 'tool_block' in interventions_dict:
            if 'size' in interventions_dict['tool_block']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block']['size']
                if 'cartesian_position' not in interventions_dict['tool_block'] and \
                        'cylindrical_position' not in interventions_dict['tool_block']:
                    cyl_pos_tool = self._stage.get_object_state('tool_block', 'cylindrical_position')
                    cyl_pos_tool[-1] = interventions_dict['tool_block']['size'][-1] / 2.0
                    interventions_dict['tool_block']['cylindrical_position'] = cyl_pos_tool
                if 'cartesian_position' not in interventions_dict['goal_block'] and \
                        'cylindrical_position' not in interventions_dict['goal_block']:
                    cyl_pos_goal = self._stage.get_object_state('goal_block', 'cylindrical_position')
                    cyl_pos_goal[-1] = interventions_dict['tool_block']['size'][-1] / 2.0
                    interventions_dict['goal_block']['cylindrical_position'] = cyl_pos_goal
        return interventions_dict

    def sample_new_goal(self, level=None):
        """
        Used to sample new goal from the corresponding shape families.

        :param level: (int) specifying the level - not used for now.

        :return: (dict) the corresponding interventions dict that could then
                       be applied to get a new sampled goal.
        """
        intervention_space = self.get_variable_space_used()
        rigid_block_side = np.random.randint(0, 2)
        # goal_block_side = not rigid_block_side
        intervention_dict = dict()
        intervention_dict['tool_block'] = dict()
        intervention_dict['goal_block'] = dict()
        if rigid_block_side == 0:
            intervention_dict['tool_block']['cylindrical_position'] = np.array(
                [0.09, -np.pi / 2, self._stage.get_object_state('tool_block', 'size')[-1] / 2.0])
        else:
            intervention_dict['tool_block']['cylindrical_position'] = np.array(
                [0.09, np.pi/2, self._stage.get_object_state('tool_block', 'size')[-1]/2.0])
        self._adjust_variable_spaces_after_intervention(intervention_dict)
        intervention_dict['goal_block']['cylindrical_position'] = \
            np.random.uniform(
                intervention_space['goal_block']['cylindrical_position'][0],
                intervention_space['goal_block']['cylindrical_position'][1])
        intervention_dict['goal_block']['euler_orientation'] = \
            np.random.uniform(intervention_space['goal_block']['euler_orientation'][0],
                              intervention_space['goal_block']['euler_orientation'][1])
        return intervention_dict

    def _adjust_variable_spaces_after_intervention(self, interventions_dict):
        """

        :param interventions_dict:
        :return:
        """
        spaces = [self._intervention_space_a,
                  self._intervention_space_b,
                  self._intervention_space_a_b]
        if 'tool_block' in interventions_dict:
            if 'cylindrical_position' in interventions_dict['tool_block']:
                if interventions_dict['tool_block']['cylindrical_position'][1] < 0:
                    tool_block_multiplier = -1
                    goal_block_multiplier = 1
                else:
                    tool_block_multiplier = 1
                    goal_block_multiplier = -1
                for variables_space in spaces:
                    if tool_block_multiplier == -1:
                        variables_space['tool_block'][
                            'cylindrical_position'][1][1] = tool_block_multiplier * \
                                                            (np.pi/6)
                        variables_space['tool_block'][
                            'cylindrical_position'][0][1] = tool_block_multiplier * \
                                                            (5/6.0) * np.pi
                        variables_space['goal_block'][
                            'cylindrical_position'][0][1] = goal_block_multiplier * \
                                                            (np.pi / 6)
                        variables_space['goal_block'][
                            'cylindrical_position'][1][1] = goal_block_multiplier * \
                                                            (5 / 6.0) * np.pi
                    else:
                        variables_space['tool_block'][
                            'cylindrical_position'][0][
                            1] = tool_block_multiplier * \
                                 (np.pi / 6)
                        variables_space['tool_block'][
                            'cylindrical_position'][1][
                            1] = tool_block_multiplier * \
                                 (5 / 6.0) * np.pi
                        variables_space['goal_block'][
                            'cylindrical_position'][1][
                            1] = goal_block_multiplier * \
                                 (np.pi / 6)
                        variables_space['goal_block'][
                            'cylindrical_position'][0][
                            1] = goal_block_multiplier * \
                                 (5 / 6.0) * np.pi
            if 'size' in interventions_dict['tool_block']:
                for variable_space in spaces:
                    variable_space['tool_block'][
                        'cylindrical_position'][0][
                        -1] = \
                        self._stage.get_object_state('tool_block', 'size')[
                            -1] / 2.0
                    variable_space['tool_block'][
                        'cylindrical_position'][
                        1][-1] = \
                        self._stage.get_object_state('tool_block', 'size')[
                            -1] / 2.0
                    variable_space['goal_block'][
                        'cylindrical_position'][0][
                        -1] = \
                        self._stage.get_object_state('goal_block', 'size')[
                            -1] / 2.0
                    variable_space['goal_block'][
                        'cylindrical_position'][
                        1][-1] = \
                        self._stage.get_object_state('goal_block', 'size')[
                            -1] / 2.0
        return



