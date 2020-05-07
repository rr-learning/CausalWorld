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
        #observation spaces are always robot obs and then stage obs
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions"]
        self.task_stage_observation_keys = ["block_position",
                                            "goal_block_position",
                                            "goal_block_orientation"]

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube",
                                            mass=0.02)
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        self.stage.finalize_stage()
        return

    def reset_task(self):
        #reset task always starts with clearing stage and robot
        self.robot.clear()
        self.stage.clear()
        sampled_positions = self.robot.sample_positions()
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))
        self.task_solved = False
        self.reset_scene_objects()
        task_observations = self.filter_observations()
        return task_observations

    def reset_scene_objects(self):
        # TODO: Refactor the orientation sampling into a general util method
        # block_position = [0.0, -0.02, 0.045155]
        # block_orientation = euler_to_quaternion([0, 0, 0.0])
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
        return

    def get_description(self):
        return \
            "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_state = self.stage.get_object_state('block')
        block_position = block_state["block_position"]
        block_orientation = block_state["block_orientation"]
        goal_state = self.stage.get_object_state('goal_block')
        goal_position = goal_state["goal_block_position"]
        goal_orientation = goal_state["goal_block_orientation"]
        position_distance = np.linalg.norm(goal_position - block_position)
        orientation_distance = np.linalg.norm(goal_orientation - block_orientation)

        reward = - position_distance - orientation_distance

        if position_distance < 0.02 and orientation_distance < 0.05:
            self.task_solved = True

        return reward

    def is_done(self):
        return self.task_solved

    def filter_observations(self):
        robot_observations_dict = self.robot.get_current_observations()
        stage_observations_dict = self.stage.get_current_observations()
        full_observations_dict = dict(robot_observations_dict)
        full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.task_robot_observation_keys:
            if key == "action_joint_positions":
                last_action_applied = self.robot.get_last_clippd_action()
                if self.robot.normalize_actions and not self.robot.normalize_observations:
                    last_action_applied = self.robot.denormalize_observation_for_key(observation=last_action_applied,
                                                                                     key='action_joint_positions')
                elif not self.robot.normalize_actions and self.robot.normalize_observations:
                    last_action_applied = self.robot.normalize_observation_for_key(observation=last_action_applied,
                                                                                     key='action_joint_positions')
                observations_filtered = \
                    np.append(observations_filtered, last_action_applied)
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(full_observations_dict[key]))

        for key in self.task_stage_observation_keys:
            observations_filtered = \
                np.append(observations_filtered,
                          np.array(full_observations_dict[key]))

        return observations_filtered

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
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet imeplemented")

    def get_task_params(self):
        task_params_dict = dict()
        return task_params_dict
