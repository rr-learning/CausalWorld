"""
This tutorial shows you how to use a handcrafted policy to solve stacking 2
task.
"""
from causal_world import CausalWorld
from causal_world import task_generator
from causal_world.actors.grasping_policy import GraspingPolicy


def example():
    task = task_generator(task_generator_id='stacking2',
                          tool_block_mass=0.02)
    env = CausalWorld(task=task, enable_visualization=True,
                      action_mode='end_effector_positions')
    policy = GraspingPolicy(tool_blocks_order=[0, 1])

    for _ in range(20):
        policy.reset()
        obs = env.reset()
        for _ in range(6000):
            obs, reward, done, info = env.step(policy.act(obs))
    env.close()


if __name__ == '__main__':
    example()
