from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np


class PickingTaskGenerator(BaseTask):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(task_name="picking",
                         use_train_space_only=kwargs.get("use_train_space_only",
                                                         True),
                         fractional_reward_weight=
                         kwargs.get("fractional_reward_weight", 1),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([250, 0, 125,
                                              0, 750, 0, 0,
                                              0.005])))
        self._task_robot_observation_keys = ["time_left_for_task",
                                            "joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions"]
        # TODO: check for nans when bounds are the same in normalization
        self._task_params["goal_height"] = \
            kwargs.get("goal_height", 0.15)
        self._task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.02)
        self._task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self._task_params["tool_block_position"] = \
            kwargs.get("tool_block_position", np.array([0, 0, 0.0325]))
        self._task_params["tool_block_orientation"] = \
            kwargs.get("tool_block_orientation", np.array([0, 0, 0, 1]))
        self.previous_object_position = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def get_description(self):
        """

        :return:
        """
        return "Task where the goal is to pick a " \
               "cube towards a goal height"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {'name': "tool_block",
                         'shape': "cube",
                         'initial_position': self._task_params
                         ["tool_block_position"],
                         'initial_orientation': self._task_params
                         ["tool_block_orientation"],
                         'mass': self._task_params["tool_block_mass"]}
        self._stage.add_rigid_general_object(**creation_dict)
        goal_block_position = np.array(
            self._task_params["tool_block_position"])
        goal_block_position[-1] = self._task_params["goal_height"]
        creation_dict = {'name': "goal_block",
                         'shape': "cube",
                         'position': goal_block_position,
                         'orientation': self._task_params
                         ["tool_block_orientation"]}
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
        super(PickingTaskGenerator, self)._set_training_intervention_spaces()
        for rigid_object in self._stage.get_rigid_objects():
            self._training_intervention_spaces[rigid_object]['cylindrical_position'][0][
                -1] \
                = 0.0325
            self._training_intervention_spaces[rigid_object]['cylindrical_position'][1][
                -1] \
                = 0.0325
        for visual_object in self._stage.get_visual_objects():
            self._training_intervention_spaces[visual_object]['cylindrical_position'][
                0][-1] \
                = 0.08
            self._training_intervention_spaces[visual_object]['cylindrical_position'][
                1][-1] \
                = 0.20
        return

    def _set_testing_intervention_spaces(self):
        """

        :return:
        """
        super(PickingTaskGenerator, self)._set_testing_intervention_spaces()
        for rigid_object in self._stage.get_rigid_objects():
            self._testing_intervention_spaces[rigid_object]['cylindrical_position'][0][
                -1] \
                = 0.0325
            self._testing_intervention_spaces[rigid_object]['cylindrical_position'][1][
                -1] \
                = 0.0325
        for visual_object in self._stage.get_visual_objects():
            self._testing_intervention_spaces[visual_object]['cylindrical_position'][0][
                -1] \
                = 0.20
            self._testing_intervention_spaces[visual_object]['cylindrical_position'][1][
                -1] \
                = 0.25
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        """

        :param desired_goal:
        :param achieved_goal:
        :return:
        """
        #rewards order
        #1) delta how much are you getting the block close to the goal
        #2) absolute how much the block is close to the goal
        #3) delta how much are you getting the block close to the center
        #4) absolute how much is the the block is close to the center
        #5) delta how much the fingers are close to block
        #6) absolute how much fingers are close to block
        #7) mean dist_of closest two fingers outside_bounding_ellipsoid
        #8) delta in joint velocities
        rewards = list()
        block_position = self._stage.get_object_state('tool_block',
                                                     'cartesian_position')
        target_height = self._stage.get_object_state('goal_block',
                                                     'cartesian_position')[-1]
        joint_velocities = self._robot.get_latest_full_state()['velocities']
        previous_block_to_goal = abs(self.previous_object_position[2] -
                                     target_height)
        current_block_to_goal = abs(block_position[2] - target_height)
        rewards.append(previous_block_to_goal - current_block_to_goal)
        rewards.append(-current_block_to_goal)
        previous_block_to_center = np.sqrt(
            (self.previous_object_position[0] ** 2 +
             self.previous_object_position[1] ** 2))
        current_block_to_center = np.sqrt((block_position[0] ** 2 +
                                           block_position[1] ** 2))
        rewards.append(previous_block_to_center - current_block_to_center)
        rewards.append(- current_block_to_center)

        end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        end_effector_positions = end_effector_positions.reshape(-1, 3)
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     block_position)
        # print("block position is ", block_position)
        # print("end effector positions ", end_effector_positions)
        previous_distance_from_block = np.linalg.norm(
            self.previous_end_effector_positions -
            self.previous_object_position)
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)
        rewards.append(- current_distance_from_block)
        #check for all the fingers if they are inside the sphere or not
        object_size = self._stage.get_object_state('tool_block',
                                                  'size')
        dist_outside_bounding_ellipsoid = np.copy(np.abs(end_effector_positions
                                                         - block_position))
        dist_outside_bounding_ellipsoid[dist_outside_bounding_ellipsoid <
                                        object_size] = 0
        dist_outside_bounding_ellipsoid = \
            np.mean(dist_outside_bounding_ellipsoid, axis=1)
        dist_outside_bounding_ellipsoid.sort()
        rewards.append(- np.sum(dist_outside_bounding_ellipsoid[:2]))
        rewards.append(- np.linalg.norm(
            joint_velocities - self.previous_joint_velocities))
        update_task_info = {
            'current_end_effector_positions': end_effector_positions,
            'current_tool_block_position': block_position,
            'current_velocity': joint_velocities}
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
        self.previous_object_position = \
            self._stage.get_object_state('tool_block', 'cartesian_position')
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

    def sample_new_goal(self, training=True, level=None):
        """

        :param training:
        :param level:
        :return:
        """
        # TODO: make sure its feasible goal by
        intervention_dict = dict()
        intervention_dict['goal_block'] = dict()
        if training:
            intervention_space = self._training_intervention_spaces
        else:
            intervention_space = self._testing_intervention_spaces
        intervention_dict['goal_block']['cylindrical_position'] = \
            np.array(self._stage.get_rigid_objects()
                     ['tool_block'].get_initial_position())
        intervention_dict['goal_block']['cylindrical_position'][-1] = \
            np.random.uniform(intervention_space['goal_block']['cylindrical_position']
                              [0][-1],
                              intervention_space['goal_block']['cylindrical_position']
                              [1][-1])
        return intervention_dict
