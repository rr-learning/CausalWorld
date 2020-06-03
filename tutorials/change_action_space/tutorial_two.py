from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.planning_wrappers import ObjectSelectorWrapper


def example():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True)
    env = ObjectSelectorWrapper(env)
    for _ in range(50):
        obs = env.reset()
        #go up
        for i in range(80):
            obs, reward, done, info = env.step([0, 1, 0])
            # print(obs)
        # rotate yaw
        for i in range(20):
            obs, reward, done, info = env.step([0, 0, 1])
            # print(obs)
        for i in range(50):
            obs, reward, done, info = env.step([0, 5, 0])
        for i in range(20):
            obs, reward, done, info = env.step([0, 0, 1])
            # print(obs)
        for i in range(50):
            obs, reward, done, info = env.step([0, 2, 0])
            # print(obs)
        print("reset")
    env.close()


if __name__ == '__main__':
    example()
