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
                                                         0)
        self.task_params["reward_weight_2"] = kwargs.get("reward_weight_2",
                                                         10)
        self.task_params["reward_weight_3"] = kwargs.get("reward_weight_3",
                                                         0)
        self.task_params["reward_weight_4"] = kwargs.get("reward_weight_4",
                                                         0)
        self.end_effector_positions_goal = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        self.stage.add_silhoutte_general_object(name="goal_1",
                                                shape="sphere",
                                                colour=np.array([1, 0, 0]))
        self.stage.add_silhoutte_general_object(name="goal_2",
                                                shape="sphere",
                                                colour=np.array([0, 1, 0]))
        self.stage.add_silhoutte_general_object(name="goal_3",
                                                shape="sphere",
                                                colour=np.array([0, 0, 1]))
        return

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
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        joints_goal = self.robot.sample_joint_positions()
        self.end_effector_positions_goal = self.robot.\
            compute_end_effector_positions(joints_goal)
        self.stage.set_objects_pose(names=["goal_1", "goal_2", "goal_3"],
                                    positions=[self.end_effector_positions_goal[:3],
                                               self.end_effector_positions_goal[3:6],
                                               self.end_effector_positions_goal[6:]],
                                    orientations=[None,
                                                  None,
                                                  None])
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
        reward_term_2 = -current_dist_to_goal
        reward_term_3 = -np.linalg.norm(self.robot.latest_full_state.torque)
        reward_term_4 = -np.linalg.norm(np.abs(
            self.robot.latest_full_state.velocity - self.previous_joint_velocities),
                                        ord=2)
        reward = self.task_params["reward_weight_1"] * reward_term_1 + \
                 self.task_params["reward_weight_2"] * reward_term_2 + \
                 self.task_params["reward_weight_3"] * reward_term_3 + \
                 self.task_params["reward_weight_4"] * reward_term_4
        self.previous_end_effector_positions = current_end_effector_positions
        self.previous_joint_velocities = np.copy(self.robot.latest_full_state.velocity)
        print(reward)
        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

