from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def test_task_sampler():
    task = Task(task_id='stacked_blocks')
    env = World(task=task, enable_visualization=True, seed=0)
    obs = env.reset()
    desired_action = env.action_space.low
    for _ in range(10000):
        obs, reward, done, info = env.step(desired_action)
    env.close()


test_task_sampler()
