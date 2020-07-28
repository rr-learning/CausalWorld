from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import quaternion_conjugate, \
    quaternion_mul
import numpy as np


class PushingTaskGenerator(BaseTask):
    def __init__(self, use_train_space_only=False,
                 fractional_reward_weight=1,
                 dense_reward_weights=np.array([750, 250, 100]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.08,
                 joint_positions=None,
                 tool_block_position=np.array([0, -0.08, 0.0325]),
                 tool_block_orientation=np.array([0, 0, 0, 1]),
                 goal_block_position=np.array([0, 0.08, 0.0325]),
                 goal_block_orientation=np.array([0, 0, 0, 1])):
        """
        This task generates a task for pushing an object on the arena's floor.

        :param use_train_space_only:
        :param fractional_reward_weight:
        :param dense_reward_weights:
        :param activate_sparse_reward:
        :param tool_block_mass:
        :param joint_positions:
        :param tool_block_position:
        :param tool_block_orientation:
        :param goal_block_position:
        :param goal_block_orientation:
        """
        super().__init__(task_name="pushing",
                         use_train_space_only=use_train_space_only,
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
        self.previous_end_effector_positions = None
        self.previous_object_position = None
        self.previous_object_orientation = None

    def get_description(self):
        """

        :return:
        """
        return \
            "Task where the goal is to push " \
            "an object towards a goal position"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {'name': "tool_block",
                         'shape': "cube",
                         'initial_position': self._task_params
                         ["tool_block_position"],
                         'initial_orientation': self._task_params
                         ["goal_block_position"],
                         'mass': self._task_params["tool_block_mass"]}
        self._stage.add_rigid_general_object(**creation_dict)
        creation_dict = {'name': "goal_block",
                         'shape': "cube",
                         'position': self._task_params
                         ["goal_block_position"],
                         'orientation':  self._task_params
                         ["goal_block_orientation"]}
        self._stage.add_silhoutte_general_object(**creation_dict)
        self._task_stage_observation_keys = ["tool_block_type",
                                            "tool_block_size",
                                            "tool_block_cartesian_position",
                                            "tool_block_orientation",
                                            "tool_block_linear_velocity",
                                            "tool_block_angular_velocity",
                                            "goal_block_type",
                                            "goal_block_size",
                                            "goal_block_cartesian_position",
                                            "goal_block_orientation"]
        return

    def _set_training_intervention_spaces(self):
        """

        :return:
        """
        super(PushingTaskGenerator, self)._set_training_intervention_spaces()
        for rigid_object in self._stage.get_rigid_objects():
            #TODO: make it a function of size
            self._training_intervention_spaces[rigid_object]['cylindrical_position'][0][-1] \
                = 0.0325
            self._training_intervention_spaces[rigid_object]['cylindrical_position'][1][-1] \
                = 0.0325
        for visual_object in self._stage.get_visual_objects():
            self._training_intervention_spaces[visual_object]['cylindrical_position'][0][-1] \
                = 0.0325
            self._training_intervention_spaces[visual_object]['cylindrical_position'][1][-1] \
                = 0.0325
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        super(PushingTaskGenerator, self)._set_testing_intervention_spaces()
        for rigid_object in self._stage.get_rigid_objects():
            self._testing_intervention_spaces[rigid_object]['cylindrical_position'][0][-1] \
                = 0.0325
            self._testing_intervention_spaces[rigid_object]['cylindrical_position'][1][-1] \
                = 0.0325
        for visual_object in self._stage.get_visual_objects():
            self._testing_intervention_spaces[visual_object]['cylindrical_position'][0][-1] \
                = 0.0325
            self._testing_intervention_spaces[visual_object]['cylindrical_position'][1][-1] \
                = 0.0325
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:

        :return:
        """
        # rewards order
        # 1) delta how much the fingers are close to block
        # 2) delta how much are you getting the block close to the goal
        # 2) delta how much the object orientation is close to goal orientation
        # 1) delta how much are you getting the block close to the goal
        # 2) absolute how much the block is close to the goal
        # 3) delta how much are you getting the block close to the center
        # 4) absolute how much is the the block is close to the center
        # 6) absolute how much fingers are close to block
        # 7) mean dist_of closest two fingers outside_bounding_ellipsoid
        # 8) delta in joint velocities
        rewards = list()
        block_position = self._stage.get_object_state('tool_block',
                                                     'cartesian_position')
        block_orientation = self._stage.get_object_state('tool_block',
                                                        'orientation')
        goal_position = self._stage.get_object_state('goal_block',
                                                    'cartesian_position')
        goal_orientation = self._stage.get_object_state('goal_block',
                                                       'orientation')
        end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
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
        previous_dist_to_goal = np.linalg.norm(goal_position -
                                               self.previous_object_position)
        current_dist_to_goal = np.linalg.norm(goal_position - block_position)
        rewards.append(previous_dist_to_goal - current_dist_to_goal)

        # calculate third reward term
        quat_diff_old = quaternion_mul(np.expand_dims(goal_orientation, 0),
                                       quaternion_conjugate(np.expand_dims(
                                           self.previous_object_orientation,
                                           0)))
        angle_diff_old = 2 * np.arccos(np.clip(quat_diff_old[:, 3], -1., 1.))

        quat_diff = quaternion_mul(np.expand_dims(goal_orientation, 0),
                                   quaternion_conjugate(np.expand_dims(
                                       block_orientation,
                                       0)))
        current_angle_diff = 2 * np.arccos(np.clip(quat_diff[:, 3], -1., 1.))

        rewards.append(angle_diff_old[0] -
                       current_angle_diff[0])
        update_task_info = {'current_end_effector_positions': end_effector_positions,
                            'current_tool_block_position': block_position,
                            'current_tool_block_orientation': block_orientation}
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
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        """

        :param interventions_dict:

        :return:
        """
        #for example size on goal_or tool should be propagated to the other
        #TODO:if a goal block intervention would lead to change of sides then
        #change the other side as well?
        if 'goal_block' in interventions_dict:
            if 'size' in interventions_dict['goal_block']:
                if 'tool_block' not in interventions_dict:
                    interventions_dict['tool_block'] = dict()
                interventions_dict['tool_block']['size'] = \
                    interventions_dict['goal_block']['size']
        elif 'tool_block' in interventions_dict:
            if 'size' in interventions_dict['tool_block']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block']['size']
        return interventions_dict

