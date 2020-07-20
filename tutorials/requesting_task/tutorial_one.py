from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.sim2real_tools.utils import RealisticRobotWrapper


def example():
    task = task_generator(task_generator_id='picking')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(20):
        env.reset()
        for _ in range(1000):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
