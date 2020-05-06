from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.state_utils import euler_to_quaternion
import numpy as np


class PushingTask(BaseTask):
    def __init__(self, task_params=None):
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
        return

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))

        self.task_solved = False
        self.reset_scene_objects()

        return self.robot.get_current_full_observations()

    def get_description(self):
        return \
            "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_state = self.stage.get_object_state('block')
        block_position = block_state["rigid_block_position"]
        block_orientation = block_state["rigid_block_orientation"]
        goal_state = self.stage.get_object_state('goal_block')
        goal_position = goal_state["silhouette_goal_block_position"]
        goal_orientation = goal_state["silhouette_goal_block_orientation"]
        position_distance = np.linalg.norm(goal_position - block_position)
        orientation_distance = np.linalg.norm(goal_orientation - block_orientation)

        reward = - position_distance - orientation_distance

        if position_distance < 0.02 and orientation_distance < 0.05:
            self.task_solved = True

        return reward

    def is_done(self):
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

    def reset_scene_objects(self):
        # TODO: Refactor the orientation sampling into a general util method

        block_position = self.stage.random_position(height_limits=0.0435)
        block_orientation = euler_to_quaternion([0, 0,
                                                 np.random.uniform(-np.pi,
                                                                   np.pi)])

        goal_position = self.stage.random_position(height_limits=0.0435)
        goal_orientation = euler_to_quaternion([0, 0,
                                                np.random.uniform(-np.pi,
                                                                  np.pi)])
        self.stage.set_objects_pose(names=["block", "goal_block"],
                                    positions=[block_position, goal_position],
                                    orientations=[block_orientation,
                                                  goal_orientation])

    def do_random_intervention(self):
        #TODO: for now just intervention on a specific object
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        # self.stage.object_intervention("block", interventions_dict)
        interventions_dict = dict()
        goal_block_position = self.stage.random_position(height_limits=0.0425)
        new_size = np.random.uniform([0.065], [0.15], size=[3,])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("goal_block", interventions_dict)
        return

    def get_task_params(self):
        # TODO: pass initialization params for this task here if we have several pushing variants in the future
        return dict()

