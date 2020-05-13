from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.delta_action_wrapper import DeltaAction
import numpy as np


def apply_delta_action():
    task = Task(task_id='reaching', randomize_joint_positions=True)
    env = World(task=task, enable_visualization=True,
                action_mode="joint_positions")
    #TODO:discuss with Manuel if the pd gains needs tuning
    env = DeltaAction(env)
    for _ in range(100):
        obs = env.reset()
        for _ in range(1000):
            obs, reward, done, info = env.step(np.zeros(9,))
    env.close()


if __name__ == '__main__':
    apply_delta_action()
