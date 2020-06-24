from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator


def example():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True,
                action_mode="joint_positions",
                normalize_observations=False,
                normalize_actions=False,
                observation_mode="cameras")
    action = env._action_space.sample()
    # new_goal = env.sample_new_goal()
    # obs = env.reset()
    # print(obs[-20:-20 + 7])
    for _ in range(10):
        obs, reward, done, info = env.step(action)
        print(obs[-20:-20 + 7])

    obs = env.reset()
    print(obs[-20:-20 + 7])
    for _ in range(10):
        obs, reward, done, info = env.step(action)
        print(obs[-20:-20 + 7])

    obs = env.reset()
    print(obs[-20:-20 + 7])
    for _ in range(2):
        obs, reward, done, info = env.step(action)
        print(obs[-20:-20 + 7])
    obs = env.reset()
    print(obs[-20:-20 + 7])
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])
    obs = env.reset()
    for _ in range(2):
        obs, reward, done, info = env.step(action)
    print(obs[-20:-20 + 7])

    obs, reward, done, info = env.step(env._action_space.sample())

    # for i in range(2000):
    #     for _ in range(200):
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         print(obs[-19:-19+7])
    #     new_goal = env.sample_new_goal()
    #     env.reset(interventions_dict=new_goal)
    env.close()


if __name__ == '__main__':
    example()
