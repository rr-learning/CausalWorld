from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator


def example():
    task = task_generator(task_generator_id='stacked_blocks')
    env = World(task=task, skip_frame=10, enable_visualization=True, seed=0,
                action_mode="joint_positions", observation_mode="cameras",
                normalize_actions=True, normalize_observations=False,
                max_episode_length=10000, enable_goal_image=True)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
