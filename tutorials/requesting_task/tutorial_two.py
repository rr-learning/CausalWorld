from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import matplotlib.pyplot as plt


def example():
    task = task_generator(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task, skip_frame=10, enable_visualization=True, seed=0,
                      action_mode="joint_positions", observation_mode="cameras")
    env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
    # Show last camera observations
    for i in range(6):
        plt.imshow(obs[i])
        plt.show()
    env.close()


if __name__ == '__main__':
    example()
