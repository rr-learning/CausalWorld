from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np


class PickingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="picking")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions",
                                            "goal_height"]
        self.task_stage_observation_keys = ["block_position"]
        self.task_params["block_mass"] = kwargs.get("block_mass", 0.02)
        self.task_params["randomize_joint_positions"] = kwargs.get(
            "randomize_joint_positions", True)
        self.task_params["randomize_block_pose"] = kwargs.get(
            "randomize_block_pose", True)
        self.task_params["goal_height"] = kwargs.get("goal_height", 0.15)
        self.task_params["reward_weight_1"] = kwargs.get("reward_weight_1", 1)
        self.task_params["reward_weight_2"] = kwargs.get("reward_weight_2", 1)
        self.task_params["reward_weight_3"] = kwargs.get("reward_weight_3", 1)
        self.previous_object_position = None

    def _set_up_non_default_observations(self):
        self._setup_non_default_robot_observation_key(
            observation_key="goal_height",
            observation_function=self._set_goal_height,
            lower_bound=[0.04], upper_bound=[0.3])
        return

    def _set_goal_height(self):
        return self.task_params["goal_height"]

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube",
                                            mass=self.task_params["block_mass"])
        self.stage.add_silhoutte_general_object(name="goal_position",
                                                shape="cube")
        return

    def _reset_task(self):
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_joint_positions()
        else:
            positions = self.robot.get_rest_pose()[0]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))

        # reset stage next
        if self.task_params["randomize_block_pose"]:
            block_position = self.stage.random_position(height_limits=0.0425)
            block_orientation = euler_to_quaternion([0, 0, np.random.uniform(0,
                                                                             np.pi)])
        else:
            block_position = [0, 0, 0.0425]
            block_orientation = euler_to_quaternion([0, 0, 0])
        goal_position = [0, 0, self.task_params["goal_height"]]
        goal_orientation = euler_to_quaternion([0, 0, 0])
        self.stage.set_objects_pose(names=["block", "goal_position"],
                                    positions=[block_position, goal_position],
                                    orientations=[block_orientation, goal_orientation])
        self.previous_object_position = block_position
        return

    def get_description(self):
        return "Task where the goal is to pick a " \
               "cube towards a goal height"

    def get_reward(self):
        # TODO: now we dont provide a structured observations for
        # the sparse reward of this task
        reward_term_1 = self._compute_sparse_reward(
            achieved_goal=None,
            desired_goal=None,
            info=self.get_info())

        block_position = self.stage.get_object_state('block', 'position')
        target_height = self.task_params["goal_height"]
        #reward term one
        previous_block_to_goal = -abs(self.previous_object_position[2] -
                                      target_height)
        current_block_to_goal = -abs(block_position[2] - target_height)
        reward_term_2 = previous_block_to_goal - current_block_to_goal

        # reward term two
        previous_block_to_center = -(self.previous_object_position[0]**2 +
                                    self.previous_object_position[1]**2)
        current_block_to_center = -(block_position[0] ** 2 +
                                    block_position[1] ** 2)
        reward_term_3 = previous_block_to_center - current_block_to_center

        reward = self.task_params["reward_weight_1"] * reward_term_1 + \
                 self.task_params["reward_weight_2"] * reward_term_2 + \
                 self.task_params["reward_weight_3"] * reward_term_3
        self.previous_object_position = block_position
        return reward

    def do_random_intervention(self):
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        new_size = np.random.uniform([0.065], [0.15], size=[3, ])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("block", interventions_dict)
        self.previous_object_position = new_block_position
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

