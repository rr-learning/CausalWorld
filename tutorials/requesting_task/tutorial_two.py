from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import matplotlib.pyplot as plt


def example():
    task = task_generator(task_generator_id='stacked_blocks')
    env = World(task=task, skip_frame=10, enable_visualization=True, seed=0,
                action_mode="joint_positions", observation_mode="cameras",
                normalize_observations=False)
    env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
    #show last images
    plt.imshow(obs[0])
    plt.show()
    plt.imshow(obs[3])
    plt.show()
    env.close()


if __name__ == '__main__':
    example()
