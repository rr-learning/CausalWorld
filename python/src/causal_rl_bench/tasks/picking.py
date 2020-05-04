from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.state_utils import euler_to_quaternion
import numpy as np


class PickingTask(Task):
    def __init__(self, task_params=None):
        super().__init__()
        self.id = "picking"
        self.robot = None
        self.stage = None
        self.observation_keys = ["joint_positions",
                                 "rigid_block_to_pick_position"]
        return

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.stage.add_rigid_general_object(name="block_to_pick",
                                            shape="cube")
        self.stage.finalize_stage()
        return

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(sampled_positions)
        new_block_position = self.stage.random_position(height_limits=0.0425)
        #TODO: sample new orientation too
        new_orientation = [0, 0, 0, 1]
        self.stage.set_positions(["block_to_pick"], [new_block_position],
                                  [new_orientation])
        return self.robot.get_current_full_observations()

    def get_description(self):
        return "Task where the goal is to pick an object and lift it as high as possible"

    def get_reward(self):
        reward = 0.0
        return reward

    def get_counterfactual_variant(self):
        pass

    def reset_scene_objects(self):
        pass

    def is_done(self):
        return False

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

    def do_random_intervention(self):
        #TODO: for now just intervention on a specific object
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_block_orientation = euler_to_quaternion(np.random.uniform([-np.pi],
                                                    [np.pi], size=[3,]))
        new_mass = np.random.uniform([0], [1], size=[1,])
        new_size = np.random.uniform([0.065], [0.15], size=[3, ])


        new_colour = np.random.uniform([0], [1], size=[3, ])
        new_linear_velocity = np.random.uniform([0], [1], size=[3, ])
        new_angular_velocity = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["linear_velocity"] = new_linear_velocity
        interventions_dict["colour"] = new_colour
        self.stage.object_intervention("block_to_pick", interventions_dict)



