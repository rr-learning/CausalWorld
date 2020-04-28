from causal_rl_bench.tasks.task import Task
import numpy as np


class PickingTask(Task):
    def __init__(self, task_params=None):
        super().__init__()
        self.id = "picking"
        self.robot = None
        self.stage = None
        return

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.stage.add_rigid_general_object(name="block_to_pick",
                                            shape="cube")
        return self.reset_task()

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(sampled_positions)
        new_block_position = self.stage.random_position(height_limits=0.0425)
        #TODO: sample new orientation too
        new_orientation = [0, 0, 0, 1]
        self.stage.set_states(["block_to_pick"], [new_block_position],
                              [new_orientation])
        return self.robot.get_current_full_observations()

    def get_description(self):
        return "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        reward = 0.0
        return reward

    def get_counterfactual_variant(self):
        pass

    def reset_scene_objects(self):
        pass

    def is_terminated(self):
        return False

    def filter_observations(self, observations_dict):
        observations_filtered = np.array([])
        for key in observations_dict.keys():
            observations_filtered = \
                np.append(observations_filtered,
                          np.array(observations_filtered[key]))
        return observations_filtered