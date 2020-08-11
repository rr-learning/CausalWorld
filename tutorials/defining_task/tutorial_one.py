"""
This tutorial shows you how to create a customized task and use all the
underlying functionalities of CausalWorld as is including reward calculation..etc
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.base_task import BaseTask
import numpy as np


class MyOwnTask(BaseTask):

    def __init__(self, **kwargs):
        super().__init__(task_name="new_task",
                         variables_space='space_a_b',
                         fractional_reward_weight=1,
                         dense_reward_weights=np.array([]))
        self._task_robot_observation_keys = [
            "time_left_for_task", "joint_positions", "joint_velocities",
            "end_effector_positions"
        ]

    #This is not even needed, it will just be an empty stage
    def _set_up_stage_arena(self):
        #NOTE: you need to add rigid objects before silhouettes for determinism (pybullet limitation)
        creation_dict = {
            'name': "tool_block",
            'filename': './assets/719.obj',
            'initial_position': [0, 0, 0.1]
        }
        self._stage.add_rigid_mesh_object(**creation_dict)
        creation_dict = {
            'name': "goal_block",
            'filename': './assets/719.obj',
            'position': [0, 0, 0.1]
        }
        self._stage.add_silhoutte_mesh_object(**creation_dict)
        self._task_stage_observation_keys = [
            "tool_block_type", "tool_block_size",
            "tool_block_cartesian_position", "tool_block_orientation",
            "tool_block_linear_velocity", "tool_block_angular_velocity",
            "goal_block_type", "goal_block_size",
            "goal_block_cylindrical_position", "goal_block_orientation"
        ]
        return


def example():
    task = MyOwnTask()
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2000):
        for _ in range(10):
            obs, reward, done, info = \
                env.step(env.action_space.sample())
        random_intervention_dict = env.do_single_random_intervention()
    env.close()


if __name__ == '__main__':
    example()
