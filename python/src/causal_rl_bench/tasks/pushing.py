from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion, quaternion_conjugate, quaternion_mul
import numpy as np


class PushingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pushing")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        self.task_stage_observation_keys = ["goal_block_position",
                                            "block_position",
                                            "block_orientation"]

        self.task_params["block_mass"] = kwargs.get("block_mass", 0.08)
        self.task_params["randomize_joint_positions"] = \
            kwargs.get("randomize_joint_positions", True)
        self.task_params["randomize_block_pose"] = \
            kwargs.get("randomize_block_pose", True)
        self.task_params["randomize_goal_block_pose"] = \
            kwargs.get("randomize_goal_block_pose", True)
        self.task_params["reward_weight_1"] = kwargs.get("reward_weight_1", 1)
        self.task_params["reward_weight_2"] = kwargs.get("reward_weight_2", 10)
        self.task_params["reward_weight_3"] = kwargs.get("reward_weight_3", 0)
        self.previous_end_effector_positions = None
        self.previous_object_position = None

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube",
                                            mass=self.task_params[
                                                "block_mass"])
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        return

    def _reset_task(self):
        #reset robot first
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_positions()
        else:
            positions = [0, -0.5, -0.6,
                         0, -0.4, -0.7,
                         0, -0.4, -0.7]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))

        # reset stage next
        #TODO: Refactor the orientation sampling into a general util method
        if self.task_params["randomize_block_pose"]:
            block_position = self.stage.random_position(height_limits=0.0425)
            block_orientation = euler_to_quaternion([0, 0,
                                                     np.random.uniform(-np.pi,
                                                                       np.pi)])
        else:
            block_position = [0.0, -0.02, 0.045155]
            block_orientation = euler_to_quaternion([0, 0, 0.0])

        if self.task_params["randomize_goal_block_pose"]:
            goal_position = self.stage.random_position(height_limits=0.0425)
            goal_orientation = euler_to_quaternion([0, 0,
                                                    np.random.uniform(-np.pi,
                                                                      np.pi)])
        else:
            goal_position = [0.04, -0.02, 0.045155]
            goal_orientation = euler_to_quaternion([0, 0, 0.0])
        self.stage.set_objects_pose(names=["block", "goal_block"],
                                    positions=[block_position, goal_position],
                                    orientations=[block_orientation,
                                                  goal_orientation])
        self.previous_end_effector_positions = self.robot.compute_end_effector_positions(self.robot.latest_full_state)
        self.previous_end_effector_positions = self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = block_position
        return

    def get_description(self):
        return \
            "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_position = self.stage.get_object_state('block', 'position')
        block_orientation = self.stage.get_object_state('block', 'orientation')
        goal_position = self.stage.get_object_state('goal_block', 'position')
        goal_orientation = self.stage.get_object_state('goal_block',
                                                       'orientation')
        end_effector_positions = self.robot.compute_end_effector_positions(
            self.robot.latest_full_state)
        end_effector_positions = end_effector_positions.reshape(-1, 3)

        #calculate first reward term
        current_distance_from_block = np.linalg.norm(end_effector_positions -
                                                     block_position)
        previous_distance_from_block = np.linalg.norm(self.previous_end_effector_positions -
                                                      self.previous_object_position)
        reward_term_1 = previous_distance_from_block - current_distance_from_block

        #calculate second reward term
        previous_dist_to_goal = np.linalg.norm(goal_position -
                                               self.previous_object_position)
        current_dist_to_goal = np.linalg.norm(goal_position - block_position)
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        # calculate third reward term
        quat_diff = quaternion_mul(np.expand_dims(goal_orientation, 0),
                                   quaternion_conjugate(np.expand_dims(
                                       block_orientation, 0)))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[:, 3], -1., 1.))
        reward_term_3 = angle_diff[0]

        #calculate final_reward
        reward = self.task_params["reward_weight_1"]*reward_term_1 + \
                 self.task_params["reward_weight_2"]*reward_term_2 \
                 + self.task_params["reward_weight_3"] * reward_term_3

        self.previous_end_effector_positions = end_effector_positions
        self.previous_object_position = block_position

        # if position_distance < 0.01:
        #     self.task_solved = True

        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        # TODO: for now just intervention on a specific object
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        # self.stage.object_intervention("block", interventions_dict)
        interventions_dict = dict()
        goal_block_position = self.stage.random_position(height_limits=0.0425)
        new_size = np.random.uniform([0.065], [0.15], size=[3, ])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("goal_block", interventions_dict)
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(self.robot.latest_full_state)
        self.previous_end_effector_positions = self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = new_block_position
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

