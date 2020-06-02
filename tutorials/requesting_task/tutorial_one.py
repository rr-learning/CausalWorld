from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator


def example():
    task = task_generator(task_generator_id='reaching', enable_visualization=True)
    env = World(task=task)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
