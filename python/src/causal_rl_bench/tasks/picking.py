from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.state_utils import euler_to_quaternion
import numpy as np


class PickingTask(BaseTask):
    def __init__(self, task_params=None):
        super().__init__()
        self.id = "pushing"
        self.robot = None
        self.stage = None
        self.task_solved = False
        #observation spaces are always robot obs and then stage obs
        self.task_robot_observation_keys = ["joint_positions",
                                            "whatever2"]
        self.task_stage_observation_keys = ["block_position",
                                            "whatever1"]
        #the helper keys are observations that are not included in the task observations but it will be needed in reward
        #calculation or new observations calculation
        self.robot_observation_helper_keys = ["joint_velocities"]
        self.stage_observation_helper_keys = ["block_linear_velocity"]

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        #first create stage
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube", mass=0.02)
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        #call finalize stage here!
        self.stage.finalize_stage()
        #add non standard observation spaces here
        self.stage.add_observation("whatever1", low_bound=np.zeros([9]),
                                   upper_bound=np.ones(9))
        self.robot.add_observation("whatever2", low_bound=np.zeros([9]),
                                   upper_bound=np.ones(9))
        return

    def reset_task(self):
        #reset task always starts with clearing stage and robot
        self.robot.clear()
        self.stage.clear()
        #set robot and scene objects
        sampled_positions = self.robot.sample_positions()
        #below configuration for initial position close to the block
        # sampled_positions = np.array([0., -0.5, -0.6,
        #                               0., -0.4, -0.7,
        #                               0., -0.4, -0.7])
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))
        self.task_solved = False
        self.reset_scene_objects()
        #get current observations
        task_observations = self.filter_observations()
        return task_observations

    def get_description(self):
        return \
            "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_state = self.stage.get_object_state('block')
        # robot_observations = self.robot.get_current_full_observations()
        block_position = block_state["block_position"]
        # end_effector_positions = robot_observations["end_effector_positions"].reshape(-1, 3)
        # distance_from_block = np.sum(
        #     (end_effector_positions - block_position) ** 2)
        # reward = - 1.3 * distance_from_block
        TARGET_HEIGHT = 0.1
        z = block_position[-1]
        x = block_position[0]
        y = block_position[1]
        reward = -abs(z - TARGET_HEIGHT) - (x ** 2 + y ** 2)
        return reward

    def is_done(self):
        return self.task_solved

    def filter_observations(self):
        robot_observations_dict = self.robot.get_current_observations(self.robot_observation_helper_keys)
        stage_observations_dict = self.stage.get_current_observations(self.stage_observation_helper_keys)
        full_observations_dict = dict(robot_observations_dict)
        full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.task_robot_observation_keys:
            # dont forget to handle non standard observation here
            if key == "whatever2":
                calculated_obs = full_observations_dict["block_linear_velocity"]
                np.append(observations_filtered,
                          np.zeros(9, ))
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(full_observations_dict[key]))

        for key in self.task_stage_observation_keys:
            if key == "whatever1":
                calculated_obs = full_observations_dict["joint_velocities"]
                np.append(observations_filtered,
                          np.zeros(9,))
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(full_observations_dict[key]))

        return observations_filtered

    def reset_scene_objects(self):
        # TODO: Refactor the orientation sampling into a general util method

        block_position = [0.0, -0.02, 0.045155]
        block_orientation = euler_to_quaternion([0, 0, 0.0])

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

