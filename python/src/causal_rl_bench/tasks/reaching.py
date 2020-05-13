from causal_rl_bench.tasks.base_task import BaseTask
import numpy as np


class ReachingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="pushing")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities"]

        self.task_params["randomize_joint_positions"] = \
            kwargs.get("randomize_joint_positions", False)
        self.task_params["reward_weight_1"] = kwargs.get("reward_weight_1", 1)
        self.task_params["reward_weight_2"] = kwargs.get("reward_weight_2", 10)
        self.task_params["reward_weight_3"] = kwargs.get("reward_weight_3", 0)

    def _reset_task(self):
        #reset robot first
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_positions()
        else:
            deg45 = np.pi / 4

            positions = [0, -deg45, -deg45]
            positions = positions * 3
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a point for each finger"

    def get_reward(self):
        reward = 0
        return reward

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

