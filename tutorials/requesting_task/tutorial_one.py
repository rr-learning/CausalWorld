from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import matplotlib.pyplot as plt
import numpy as np


def example():
    task = task_generator(task_generator_id='reaching')
    env = World(task=task, enable_visualization=True,
                action_mode="joint_positions",
                normalize_observations=False,
                normalize_actions=False,
                skip_frame=10)
    horizon = 100
    actions = [env.action_space.sample() for _ in range(horizon)]
    # new_goal = env.sample_new_goal()
    # env.reset(interventions_dict=new_goal)
    # env.reset()
    # obs = env.reset(interventions_dict=new_goal)
    # print(obs[-20:-20 + 7])
    # env.do_intervention({'goal_block': {'color': np.array([0, 0, 0])}})
    print("first")
    for i in range(100):
        obs, reward, done, info = env.step(actions[i])
        print(reward)
        # plt.imshow(obs[0])
        # plt.show()
        # plt.imshow(obs[3])
        # plt.show()
        #
        # print(obs[-20:-20 + 7])
    print("second")
    obs = env.reset()
    # print(obs[-20:-20 + 7])
    for i in range(100):
        obs, reward, done, info = env.step(actions[i])
        print(reward)
        # print(obs[-20:-20 + 7])

    # obs = env.reset()
    # # print(obs[-20:-20 + 7])
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    #     # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # # print(obs[-20:-20 + 7])
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # print(obs[-20:-20 + 7])
    # obs = env.reset()
    # for _ in range(2):
    #     obs, reward, done, info = env.step(action)
    # print(obs[-20:-20 + 7])
    #
    # obs, reward, done, info = env.step(env.action_space.sample())
    #
    # # for i in range(2000):
    # #     for _ in range(200):
    # #         obs, reward, done, info = env.step(env.action_space.sample())
    # #         print(obs[-19:-19+7])
    # #     new_goal = env.sample_new_goal()
    # #     env.reset(interventions_dict=new_goal)
    # env.close()


if __name__ == '__main__':
    example()
