from causal_rl_bench.task_generators.base_task import BaseTask
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
                                    np.array([1, 0.5, 3, 0.001])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        self.task_stage_observation_keys = ["tool_block_position",
                                            "tool_block_orientation",
                                            "goal_block_position",
                                            "goal_block_orientation"]
        # TODO: check for nans when bounds are the same in normalization
        self.task_params["goal_height"] = \
            kwargs.get("goal_height", 0.15)
        self.task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.02)
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.initial_state["tool_block"] = dict()
        self.initial_state["tool_block"]["position"] = \
            kwargs.get("tool_block_position", np.array([0, 0, 0.0425]))
        self.initial_state["tool_block"]["orientation"] = \
            kwargs.get("tool_block_orientation", np.array([0, 0, 0, 1]))
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
        goal_block_position = np.array(
            self.initial_state["tool_block"]["position"])
        goal_block_position[-1] = self.task_params["goal_height"]
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube",
                                                position=goal_block_position,
                                                orientation=
                                                self.initial_state[
                                                    "tool_block"][
                                                    "orientation"])
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
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
        for visual_object in self.stage.visual_objects:
            self.training_intervention_spaces[visual_object]['position'][
                0][-1] \
                = 0.08
            self.training_intervention_spaces[visual_object]['position'][
                1][-1] \
                = 0.20
        self.training_intervention_spaces['goal_height'] = \
            np.array([0.08, 0.20])
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
        for visual_object in self.stage.visual_objects:
            self.testing_intervention_spaces[visual_object]['position'][0][
                -1] \
                = 0.20
            self.testing_intervention_spaces[visual_object]['position'][1][
                -1] \
                = 0.25
        self.testing_intervention_spaces['goal_height'] = \
            np.array([0.20, 0.25])
        return

    def _calculate_dense_rewards(self, desired_goal, achieved_goal):
        rewards = list()
        block_position = self.stage.get_object_state('tool_block',
                                                     'position')
        target_height = self.stage.get_object_state('goal_block',
                                                     'position')[-1]
        joint_velocities = self.robot.latest_full_state.velocity
        previous_block_to_goal = abs(self.previous_object_position[2] -
                                     target_height)
        current_block_to_goal = abs(block_position[2] - target_height)
        rewards.append(previous_block_to_goal - current_block_to_goal)
        previous_block_to_center = np.sqrt(
            (self.previous_object_position[0] ** 2 +
             self.previous_object_position[1] ** 2))
        current_block_to_center = np.sqrt((block_position[0] ** 2 +
                                           block_position[1] ** 2))
        rewards.append(previous_block_to_center - current_block_to_center)

        end_effector_positions = self.robot.compute_end_effector_positions(
            self.robot.latest_full_state.position)
        end_effector_positions = end_effector_positions.reshape(-1, 3)
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

    def sample_new_goal(self, training=True):
        # TODO: make sure its feasible goal by
        intervention_dict = dict()
        if training:
            intervention_space = self.training_intervention_spaces
        else:
            intervention_space = self.testing_intervention_spaces
        intervention_dict['goal_height'] = np.\
            random.uniform(intervention_space['goal_height'][0],
                           intervention_space['goal_height'][1])
        return intervention_dict

    def get_task_generator_variables_values(self):
        return {'goal_height': self.stage.get_object_state('goal_block',
                                                           'position')[-1]}

    def apply_task_generator_interventions(self, interventions_dict):
        reset_observation_space = False
        new_interventions_dict = dict()
        for intervention_variable in interventions_dict:
            if intervention_variable == "goal_height":
                new_interventions_dict['goal_block'] = dict()
                new_interventions_dict['goal_block']['position'] = \
                    self.stage.get_object_state\
                    ('tool_block', 'position')
                new_interventions_dict['goal_block']['orientation'] = \
                    self.stage.get_object_state\
                    ('tool_block', 'orientation')
                new_interventions_dict['goal_block']['position'][-1] = \
                    interventions_dict["goal_height"]
            else:
                raise Exception("this task generator variable "
                                "is not yet defined")
        if len(new_interventions_dict) == 0:
            return True, reset_observation_space
        else:
            self.stage.apply_interventions(new_interventions_dict)
        return True, reset_observation_space
