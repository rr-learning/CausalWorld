from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def example():
    task = Task(task_id='pushing')
    env = World(task=task, enable_visualization=True, skip_frame=10)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
