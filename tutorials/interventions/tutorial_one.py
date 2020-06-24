from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator


def example():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True)
    env.reset()
    for _ in range(50):
        for i in range(10):
            obs, reward, done, info = env.step(env.action_space.sample())
            random_intervention_dict = env.do_single_random_intervention()
        print("resetting")
        env.reset()
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        print("The random intervention performed is ", random_intervention_dict)
    env.close()


if __name__ == '__main__':
    example()
