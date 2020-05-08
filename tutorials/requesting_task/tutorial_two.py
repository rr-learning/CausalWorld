from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def example():
    task = Task(task_id='cuboid_silhouette')
    env = World(task=task, skip_frame=35, enable_visualization=True, seed=0,
                action_mode="joint_positions", observation_mode="cameras",
                normalize_actions=True, normalize_observations=False,
                max_episode_length=10000, enable_goal_image=True)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
