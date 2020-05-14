from causal_rl_bench.tasks.base_task import BaseTask
import numpy as np


class ReachingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="reaching")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions",
                                            "action_joint_positions",
                                            "end_effector_positions_goal"]

        self.task_params["randomize_joint_positions"] = \
            kwargs.get("randomize_joint_positions", True)
        #TODO: implement non randomization of goal
        self.task_params["reward_weight_1"] = kwargs.get("reward_weight_1",
                                                         1)
        self.end_effector_positions_goal = None

    def _set_up_non_default_observations(self):
        self._setup_non_default_robot_observation_key(
            observation_key="end_effector_positions_goal",
            observation_function=self._set_end_effector_positions_goal,
            lower_bound=[-0.5, -0.5, 0]*3,
            upper_bound=[0.5, 0.5, 0.5]*3)
        return

    def _set_end_effector_positions_goal(self):
        return self.end_effector_positions_goal

    def _reset_task(self):
        #reset robot first
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_joint_positions()
        else:
            positions = self.robot.get_rest_pose()[0]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        joints_goal = self.robot.sample_joint_positions()
        self.end_effector_positions_goal = self.robot.\
            compute_end_effector_positions(joints_goal)
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a point for each finger"

    def get_reward(self):
        current_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        previous_dist_to_goal = np.linalg.norm(
            self.end_effector_positions_goal -
            self.previous_end_effector_positions)
        current_dist_to_goal = np.linalg.norm(self.end_effector_positions_goal
                                              - current_end_effector_positions)
        reward_term_1 = previous_dist_to_goal - current_dist_to_goal
        reward = self.task_params["reward_weight_1"] * reward_term_1
        self.end_effector_positions_goal = current_end_effector_positions
        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

