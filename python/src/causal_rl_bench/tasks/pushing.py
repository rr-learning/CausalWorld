from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.state_utils import euler_to_quaternion
import numpy as np


class PushingTask(Task):
    def __init__(self):
        super().__init__()
        self.id = "pushing"
        self.robot = None
        self.stage = None
        self.seed = 0

        self.task_solved = False

        self.observation_keys = ["end_effector_positions",
                                 "block_position"]

    def init_task(self, robot, stage):
        self.robot = robot
        self.robot.add_end_effector_postions_observations()
        self.stage = stage
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube", mass=0.005)
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

    def get_task_params(self):
        task_params_dict = dict()
        task_params_dict["task_id"] = self.id
        task_params_dict["skip_frame"] = self.robot.get_skip_frame()
        task_params_dict["seed"] = self.seed
        task_params_dict["action_mode"] = self.robot.get_action_mode()
        task_params_dict["observation_mode"] = self.robot.get_observation_mode()
        task_params_dict["camera_skip_frame"] = \
            self.robot.get_camera_skip_frame()
        task_params_dict["normalize_actions"] = \
            self.robot.robot_actions.is_normalized()
        task_params_dict["normalize_observations"] = \
            self.robot.robot_observations.is_normalized()
        task_params_dict["max_episode_length"] = None
        return task_params_dict

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


