from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.base_task import BaseTask
import numpy as np


class MyOwnTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="new_task",
                         intervention_split=False,
                         training=True,
                         sparse_reward_weight=1,
                         dense_reward_weights=np.array([]))

    def _set_up_stage_arena(self):
        self.stage.add_rigid_mesh_object('tool_block',
                                         filename='./assets/719.obj')
        self.stage.add_silhoutte_mesh_object('goal_block',
                                             filename='./assets/719.obj')
        self.task_stage_observation_keys = ["tool_block_position",
                                            "tool_block_orientation",
                                            "goal_block_position"]


def example():
    task = MyOwnTask()
    env = World(task=task, enable_visualization=True)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = \
            env.step(env.action_space.sample())
        random_intervention_dict = env.do_single_random_intervention()
    env.close()


if __name__ == '__main__':
    example()
