from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def test_mass():
    task = Task(task_id='picking', randomize_joint_positions=False,
                randomize_block_pose=False, block_mass=0.02)
    env = World(task=task, skip_frame=1, enable_visualization=True, seed=0,
                action_mode="end_effector_positions",
                observation_mode="structured",
                normalize_actions=False,
                normalize_observations=False,
                max_episode_length=10000)
    obs = env.reset()
    desired_action = obs[27:27+9]
    desired_action[:2] = [0, 0.03]
    desired_action[3:5] = [0, -0.03]
    desired_action[2] = 0.05
    desired_action[5] = 0.05
    desired_action[-1] = 0.05
    #grasp the block now
    for _ in range(250):
        obs, reward, done, info = env.step(desired_action)

    #NOW lets move up a bit by bit (1 cm each second?)
    for _ in range(40):
        desired_action[2] += 0.01
        desired_action[5] += 0.01
        for _ in range(250):
            obs, reward, done, info = env.step(desired_action)

    env.close()


test_mass()
