from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np


class ExampleTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="example_task")
        self.task_robot_observation_keys = ["joint_positions",
                                            "whatever2"]
        self.task_stage_observation_keys = ["block_position",
                                            "whatever1"]
        self._robot_observation_helper_keys = ["joint_velocities"]
        self._stage_observation_helper_keys = ["block_linear_velocity"]
        #define task params below
        self.task_params["variable_1"] = 0

    def _set_up_stage_arena(self):
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube", mass=0.02)
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        return

    def _set_up_non_default_observations(self):
        self._setup_non_default_robot_observation_key(observation_key="whatever2",
                                                      observation_function=self._calculate_whatever2_observation,
                                                      lower_bound=np.zeros([9]),
                                                      upper_bound=np.ones(9))
        self._setup_non_default_stage_observation_key(observation_key="whatever1",
                                                      observation_function=self._calculate_whatever1_observation,
                                                      lower_bound=np.zeros([9]),
                                                      upper_bound=np.ones(9))
        return

    def _calculate_whatever2_observation(self):
        calculated_obs = self.current_full_observations_dict["block_linear_velocity"]
        return np.zeros(9, )

    def _calculate_whatever1_observation(self):
        calculated_obs = self.current_full_observations_dict["joint_velocities"]
        return np.zeros(9, )

    def _reset_task(self):
        #reset robot first
        sampled_positions = self.robot.sample_joint_positions()
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))

        # reset stage next
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

        return

    def get_description(self):
        return "example task"

    def get_reward(self):
        block_position = self.stage.get_object_state('block', 'position')
        TARGET_HEIGHT = 0.1
        z = block_position[-1]
        x = block_position[0]
        y = block_position[1]
        reward = -abs(z - TARGET_HEIGHT) - (x ** 2 + y ** 2)
        return reward

    def is_done(self):
        #test of task is solved or not?
        return self.task_solved

    def do_single_random_intervention(self):
        #TODO: for now just intervention on a specific object
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_color = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["color"] = new_color
        # self.stage.object_intervention("block", interventions_dict)
        interventions_dict = dict()
        goal_block_position = self.stage.random_position(height_limits=0.0425)
        new_size = np.random.uniform([0.065], [0.15], size=[3,])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("goal_block", interventions_dict)
        return

    def do_intervention(self, **kwargs):
        pass

