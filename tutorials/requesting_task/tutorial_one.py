from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.env_wrappers import HERGoalEnvWrapper


def example():
    task = task_generator(task_generator_id='pushing')
    env = World(task=task, enable_visualization=True)
    env = HERGoalEnvWrapper(env)
    for _ in range(200):
        obs = env.reset()
        new_goal = env.sample_new_goal()
        # print(obs)
        # chosen_intervention = env.do_single_random_intervention()
        # print(chosen_intervention)
        # print("intervened")
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
            # print(obs[-9:])
            # print(reward)
            # print(done)
            # print(info)
    env.close()


if __name__ == '__main__':
    example()
