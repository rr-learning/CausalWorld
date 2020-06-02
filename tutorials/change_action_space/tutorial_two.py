from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.planning_wrappers import ObjectSelectorWrapper


def example():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True)
    env = ObjectSelectorWrapper(env)
    for _ in range(50):
        obs = env.reset()
        for i in range(2000):
            obs, reward, done, info = env.step([0, 1, 0])
            # print(obs)
    env.close()


if __name__ == '__main__':
    example()
