from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def example():
    task = Task(task_id='pick_and_place')
    env = World(task=task, enable_visualization=True)
    for _ in range(200):
        env.reset()
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
