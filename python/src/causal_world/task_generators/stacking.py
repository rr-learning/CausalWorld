from causal_world.task_generators.base_task import BaseTask
import numpy as np
import copy


class Stacking2TaskGenerator(BaseTask):
    def __init__(self, variables_space='space_a_b',
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([750, 250,
                                                250, 125,
                                                0.005]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.02,
                 joint_positions=None,
                 tool_block_1_position=np.array([0, 0, 0.0325]),
                 tool_block_1_orientation=np.array([0, 0, 0, 1]),
                 tool_block_2_position=np.array([0.07, 0.07, 0.0325]),
                 tool_block_2_orientation=np.array([0, 0, 0, 1])):
        """
        This task generates a task for picking an object in the air.

        :param variables_space:
        :param fractional_reward_weight:
        :param dense_reward_weights:
        :param activate_sparse_reward:
        :param tool_block_mass:
        :param joint_positions:
        :param tool_block_1_position:
        :param tool_block_1_orientation:
        :param tool_block_2_position:
        :param tool_block_2_orientation:
        """
        super().__init__(task_name="stacking2",
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
        self._task_params["tool_block_1_position"] = tool_block_1_position
        self._task_params["tool_block_1_orientation"] = tool_block_1_orientation
        self._task_params["tool_block_2_position"] = tool_block_2_position
        self._task_params["tool_block_2_orientation"] = tool_block_2_orientation
        self.previous_tool_block_1_position = None
        self.previous_tool_block_2_position = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def get_description(self):
        """

        :return:
        """
        return "Task where the goal shape is a tower of two blocks"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {
            'name': "tool_block_1",
            'shape': "cube",
            'initial_position': self._task_params["tool_block_1_position"],
            'initial_orientation': self._task_params["tool_block_1_orientation"],
            'mass': self._task_params["tool_block_mass"]
        }
        self._stage.add_rigid_general_object(**creation_dict)

        creation_dict = {
            'name': "tool_block_2",
            'shape': "cube",
            'initial_position': np.array([0.01, 0.08, 0.0325]),
            'initial_orientation': self._task_params["tool_block_2_orientation"],
            'mass': self._task_params["tool_block_mass"]
        }
        self._stage.add_rigid_general_object(**creation_dict)

        goal_block_1_position = np.array(self._task_params["tool_block_1_position"])
        goal_block_1_position[0] = -0.06
        goal_block_1_position[1] = -0.06

        creation_dict = {
            'name': "goal_block_1",
            'shape': "cube",
            'position': goal_block_1_position,
            'orientation': self._task_params["tool_block_1_orientation"]
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        goal_block_2_position = copy.deepcopy(np.array(self._task_params["tool_block_1_position"]))
        goal_block_2_position[0] = -0.06
        goal_block_2_position[1] = -0.06
        goal_block_2_position[2] += 0.065
        creation_dict = {
            'name': "goal_block_2",
            'shape': "cube",
            'position': goal_block_2_position,
            'orientation': self._task_params["tool_block_1_orientation"]
        }
        self._stage.add_silhoutte_general_object(**creation_dict)
        self._task_stage_observation_keys = [
            "tool_block_1_type", "tool_block_1_size",
            "tool_block_1_cartesian_position", "tool_block_1_orientation",
            "tool_block_1_linear_velocity", "tool_block_1_angular_velocity",
            "tool_block_2_type", "tool_block_2_size",
            "tool_block_2_cartesian_position", "tool_block_2_orientation",
            "tool_block_2_linear_velocity", "tool_block_2_angular_velocity",
            "goal_block_1_type", "goal_block_1_size",
            "goal_block_1_cartesian_position", "goal_block_1_orientation",
            "goal_block_2_type", "goal_block_2_size",
            "goal_block_2_cartesian_position", "goal_block_2_orientation"
        ]

        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        rewards = [0.0] * 5

        block_position_1 = self._stage.get_object_state('tool_block_1',
                                                        'cartesian_position')
        block_position_2 = self._stage.get_object_state('tool_block_2',
                                                        'cartesian_position')
        goal_block_1_position = self._stage.get_object_state('goal_block_1',
                                                             'cartesian_position')
        goal_block_2_position = self._stage.get_object_state('goal_block_2',
                                                             'cartesian_position')
        joint_velocities = self._robot.get_latest_full_state()['velocities']
        end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        end_effector_positions = end_effector_positions.reshape(-1, 3)

        lower_block_positioned = False
        if np.linalg.norm(block_position_1 - goal_block_1_position) < 0.02:
            lower_block_positioned = True

        if not lower_block_positioned:
            current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                         block_position_1)
            previous_distance_from_block = np.linalg.norm(
                self.previous_end_effector_positions -
                self.previous_tool_block_1_position)
            rewards[0] = previous_distance_from_block - current_distance_from_block

            previous_dist_to_goal = np.linalg.norm(goal_block_1_position -
                                                   self.previous_tool_block_1_position)
            current_dist_to_goal = np.linalg.norm(goal_block_1_position - block_position_1)
            rewards[1] = previous_dist_to_goal - current_dist_to_goal

        else:
            current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                         block_position_2)
            previous_distance_from_block = np.linalg.norm(
                self.previous_end_effector_positions -
                self.previous_tool_block_2_position)
            rewards[0] = previous_distance_from_block - current_distance_from_block

            block_2_above_block_1 = False
            if np.linalg.norm(block_position_1[:2] - block_position_2[:2]) < 0.005:
                block_2_above_block_1 = True

            previous_block_to_goal_height = abs(self.previous_tool_block_2_position[2] -
                                                goal_block_2_position[2])
            current_block_to_goal_height = abs(block_position_2[2] - goal_block_2_position[2])
            if not block_2_above_block_1:
                rewards[2] = (previous_block_to_goal_height -
                              current_block_to_goal_height)
            else:
                rewards[2] = 0.0

            if block_position_2[2] > goal_block_2_position[2]:
                # if block 2 high enough activate horizontal reward
                previous_block_1_to_block_2 = np.linalg.norm(
                    self.previous_tool_block_1_position[:2] -
                    self.previous_tool_block_2_position[:2])
                current_block_1_to_block_2 = np.linalg.norm(
                    block_position_1[:2] -
                    block_position_2[:2])
                rewards[3] = previous_block_1_to_block_2 - current_block_1_to_block_2
            else:
                rewards[3] = 0.0

        rewards[4] = -np.linalg.norm(joint_velocities -
                                     self.previous_joint_velocities)
        update_task_info = {
            'current_end_effector_positions': end_effector_positions,
            'current_tool_block_1_position': block_position_1,
            'current_tool_block_2_position': block_position_2,
            'current_velocity': joint_velocities
        }
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        """

        :param update_task_info:

        :return:
        """
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_tool_block_1_position = \
            update_task_info['current_tool_block_1_position']
        self.previous_tool_block_2_position = \
            update_task_info['current_tool_block_2_position']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return

    def _set_task_state(self):
        """

        :return:
        """
        self.previous_end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_tool_block_1_position = \
            self._stage.get_object_state('tool_block_1', 'cartesian_position')
        self.previous_tool_block_2_position = \
            self._stage.get_object_state('tool_block_2', 'cartesian_position')
        self.previous_joint_velocities = \
            self._robot.get_latest_full_state()['velocities']
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        """

        :param interventions_dict:

        :return:
        """
        # for example size on goal_or tool should be propagated to the other
        if 'goal_block' in interventions_dict:
            if 'size' in interventions_dict['goal_block']:
                if 'tool_block_1' not in interventions_dict:
                    interventions_dict['tool_block_1'] = dict()
                interventions_dict['tool_block_1']['size'] = \
                    interventions_dict['goal_block']['size']
                if 'tool_block_2' not in interventions_dict:
                    interventions_dict['tool_block_2'] = dict()
                interventions_dict['tool_block_2']['size'] = \
                    interventions_dict['goal_block']['size']
        elif 'tool_block_1' in interventions_dict:
            if 'size' in interventions_dict['tool_block_1']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block_1']['size']
        elif 'tool_block_2' in interventions_dict:
            if 'size' in interventions_dict['tool_block_2']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block_2']['size']
        return interventions_dict

    def sample_new_goal(self, level=None):
        """
        :param level:

        :return:
        """
        intervention_space = self.get_variable_space_used()
        pos_low_bound = np.array(intervention_space['goal_block_1']['cylindrical_position'][0])
        pos_low_bound[-1] = self._stage.get_object_state('tool_block_1', 'size')[-1] / 2.0
        pos_upper_bound = np.array(intervention_space['goal_block_1']['cylindrical_position'][1])
        pos_upper_bound[-1] = self._stage.get_object_state('tool_block_1', 'size')[-1] / 2.0
        intervention_dict = dict()
        intervention_dict['goal_block_1'] = dict()
        intervention_dict['goal_block_1']['cylindrical_position'] = \
            np.random.uniform(pos_low_bound,
                              pos_upper_bound)
        intervention_dict['goal_block_1']['euler_orientation'] = \
            np.random.uniform(intervention_space['goal_block_1']['euler_orientation'][0],
                              intervention_space['goal_block_1']['euler_orientation'][1])
        intervention_dict['goal_block_2'] = copy.deepcopy(intervention_dict['goal_block_1'])
        intervention_dict['goal_block_2']['cylindrical_position'][2] += 0.065

        return intervention_dict
