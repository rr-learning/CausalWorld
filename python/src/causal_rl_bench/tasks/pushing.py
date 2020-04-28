from counterfactual.python.src.causal_rl_bench.tasks.task import Task
from scipy.spatial.transform import Rotation as rotation
import numpy as np


class PushingTask(Task):
    def __init__(self):
        super().__init__()
        self.id = "pushing"
        self.robot = None
        self.stage = None

        self.task_solved = False

        self.observation_keys = ["joint_positions",
                                 "rigid_block_position",
                                 "rigid_block_orientation",
                                 "silhouette_goal_block_position",
                                 "silhouette_goal_block_orientation"]

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube")
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        self.stage.finalize_stage()

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(sampled_positions)

        self.task_solved = False

        # TODO: Refactor the orientation sampling into a general util method

        block_position = self.stage.random_position(height_limits=0.0435)
        block_orientation = rotation.from_euler('z', np.random.uniform(0, 360),
                                                degrees=True).as_quat()

        goal_position = self.stage.random_position(height_limits=0.0435)
        goal_orientation = rotation.from_euler('z', np.random.uniform(0, 360),
                                               degrees=True).as_quat()

        self.stage.set_states(names=["block", "goal_block"],
                              positions=[block_position, goal_position],
                              orientations=[block_orientation, goal_orientation])
        return self.robot.get_current_full_observations()

    def get_description(self):
        return "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        state = self.stage.get_full_state()
        block_position = state["rigid_block_position"]
        block_orientation = state["rigid_block_orientation"]
        goal_position = state["silhouette_goal_block_position"]
        goal_orientation = state["silhouette_goal_block_orientation"]
        position_distance = np.linalg.norm(goal_position - block_position)
        orientation_distance = np.linalg.norm(goal_orientation - block_orientation)

        reward = - position_distance - orientation_distance

        if position_distance < 0.02 and orientation_distance < 0.05:
            self.task_solved = True

        return reward

    def is_terminated(self):
        return self.task_solved

    def filter_observations(self, robot_observations_dict,
                            stage_observations_dict):
        full_observations_dict = dict(robot_observations_dict)
        full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.observation_keys:
            observations_filtered = \
                np.append(observations_filtered,
                          np.array(full_observations_dict[key]))
        return observations_filtered

    def get_counterfactual_variant(self):
        pass

    def reset_scene_objects(self):
        pass
