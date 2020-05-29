from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np


class PickingTaskGenerator(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="picking",
                         intervention_split=kwargs.get(
                             "intervention_split",
                             False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 0),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([1000, 500, 3000, 1])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        self.task_stage_observation_keys = ["tool_block_position",
                                            "tool_block_orientation",
                                            "goal_block_position",
                                            "goal_block_orientation"]
        self.task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.02)
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.initial_state["tool_block"] = dict()
        self.initial_state["tool_block"]["position"] = \
            kwargs.get("tool_block_position", np.array([0, 0, 0.0425]))
        self.initial_state["tool_block"]["orientation"] = \
            kwargs.get("tool_block_orientation", np.array([0, 0, 0, 1]))
        self.task_params["goal_block_position"] = \
            kwargs.get("goal_block_position", np.array([0, 0., 0.15]))
        self.task_params["goal_block_orientation"] = \
            kwargs.get("goal_block_orientation", np.array([0, 0, 0, 1]))
        self.previous_object_position = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def get_description(self):
        return "Task where the goal is to pick a " \
               "cube towards a goal height"

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="tool_block",
                                            shape="cube",
                                            mass=self.task_params[
                                                "tool_block_mass"],
                                            position=self.initial_state
                                            ["tool_block"]["position"],
                                            orientation=self.initial_state
                                            ["tool_block"]["orientation"])
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube",
                                                position=self.task_params[
                                                    "goal_block_position"],
                                                orientation=
                                                self.task_params[
                                                    "goal_block_orientation"])
        return

    def _set_training_intervention_spaces(self):
        super(PickingTaskGenerator, self)._set_training_intervention_spaces()
        for rigid_object in self.stage.rigid_objects:
            self.training_intervention_spaces[rigid_object]['position'][0][
                -1] \
                = 0.0425
            self.training_intervention_spaces[rigid_object]['position'][1][
                -1] \
                = 0.0425
            self.training_intervention_spaces[rigid_object]['position'][0][0] \
                += 0.04
            self.training_intervention_spaces[rigid_object]['position'][0][1] \
                += 0.04
            self.training_intervention_spaces[rigid_object]['position'][1][0] \
                -= 0.04
            self.training_intervention_spaces[rigid_object]['position'][1][1] \
                -= 0.04
        for visual_object in self.stage.visual_objects:
            self.training_intervention_spaces[visual_object]['position'][
                0][-1] \
                = 0.08
            self.training_intervention_spaces[visual_object]['position'][
                1][-1] \
                = 0.20
            self.training_intervention_spaces[visual_object]['position'][0][0] \
                += 0.04
            self.training_intervention_spaces[visual_object]['position'][0][1] \
                += 0.04
            self.training_intervention_spaces[visual_object]['position'][1][0] \
                -= 0.04
            self.training_intervention_spaces[visual_object]['position'][1][1] \
                -= 0.04
        return

    def _set_testing_intervention_spaces(self):
        super(PickingTaskGenerator, self)._set_testing_intervention_spaces()
        for rigid_object in self.stage.rigid_objects:
            self.testing_intervention_spaces[rigid_object]['position'][0][
                -1] \
                = 0.0425
            self.testing_intervention_spaces[rigid_object]['position'][1][
                -1] \
                = 0.0425
            self.testing_intervention_spaces[rigid_object]['position'][0][0] \
                += 0.04
            self.testing_intervention_spaces[rigid_object]['position'][0][1] \
                += 0.04
            self.testing_intervention_spaces[rigid_object]['position'][1][0] \
                -= 0.04
            self.testing_intervention_spaces[rigid_object]['position'][1][1] \
                -= 0.04
        for visual_object in self.stage.visual_objects:
            self.testing_intervention_spaces[visual_object]['position'][0][
                -1] \
                = 0.20
            self.testing_intervention_spaces[visual_object]['position'][1][
                -1] \
                = 0.25
            self.testing_intervention_spaces[visual_object]['position'][0][0] \
                += 0.04
            self.testing_intervention_spaces[visual_object]['position'][0][1] \
                += 0.04
            self.testing_intervention_spaces[visual_object]['position'][1][0] \
                -= 0.04
            self.testing_intervention_spaces[visual_object]['position'][1][1] \
                -= 0.04
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        rewards = list()
        block_position = self.stage.get_object_state('tool_block',
                                                     'position')
        goal_position = self.stage.get_object_state('goal_block',
                                                    'position')
        target_height = goal_position[-1]
        joint_velocities = self.robot.latest_full_state.velocity
        # reward term one
        previous_block_to_goal = abs(self.previous_object_position[2] -
                                     target_height)
        current_block_to_goal = abs(block_position[2] - target_height)
        rewards.append(previous_block_to_goal - current_block_to_goal)

        # reward term two
        previous_block_to_center = np.sqrt(
            (self.previous_object_position[0] ** 2 +
             self.previous_object_position[1] ** 2))
        current_block_to_center = np.sqrt((block_position[0] ** 2 +
                                           block_position[1] ** 2))
        rewards.append(previous_block_to_center - current_block_to_center)

        end_effector_positions = self.robot.compute_end_effector_positions(
            self.robot.latest_full_state.position)
        end_effector_positions = end_effector_positions.reshape(-1, 3)

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     block_position)
        previous_distance_from_block = np.linalg.norm(
            self.previous_end_effector_positions -
            self.previous_object_position)
        rewards.append(previous_distance_from_block -
                       current_distance_from_block)

        rewards.append(- np.linalg.norm(
            joint_velocities - self.previous_joint_velocities))
        update_task_info = {
            'current_end_effector_positions': end_effector_positions,
            'current_tool_block_position': block_position,
            'current_velocity': joint_velocities}
        return rewards, update_task_info

    def _update_task_state(self, update_task_info):
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_object_position = \
            update_task_info['current_tool_block_position']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return

    def _set_task_state(self):
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = \
            self.stage.get_object_state('tool_block', 'position')
        self.previous_joint_velocities = \
            self.robot.latest_full_state.velocity
        return

    def _handle_contradictory_interventions(self, interventions_dict):
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

    def sample_new_goal(self):
        # TODO: make sure its feasible goal by
        # taking care of the size as well
        lower_bound = self.stage.floor_inner_bounding_box[0]
        upper_bound = self.stage.floor_inner_bounding_box[1]
        lower_bound[-1] = 0.07
        upper_bound[-1] = 0.22
        new_goal = dict()
        new_goal['goal_block'] = dict()
        new_goal['goal_block']['position'] \
            = np.random.uniform(lower_bound, upper_bound)
        new_goal['goal_block']['orientation'] \
            = euler_to_quaternion([0, 0, 0])
        return new_goal


