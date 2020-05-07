from causal_rl_bench.tasks.task import Task
from causal_rl_bench.envs.world import World

import numpy as np


def test_determinism():
    task = Task(task_id="cuboid_silhouette")
    observations_v1 = []
    observations_v2 = []
    observations_v3 = []
    rewards_v1 = []
    rewards_v2 = []
    rewards_v3 = []
    horizon = 30

    env_v1 = World(task=task,
                   enable_visualization=False,
                   seed=27)

    obs = env_v1.reset()
    observations_v1.append(obs)
    for _ in range(horizon):
        obs, reward, done, info = env_v1.step(env_v1.action_space.low)
        observations_v1.append(obs)
        rewards_v1.append(reward)
    env_v1.close()

    env_v2 = World(task=task,
                   enable_visualization=False,
                   seed=27)

    obs = env_v2.reset()
    observations_v2.append(obs)
    for _ in range(horizon):
        obs, reward, done, info = env_v2.step(env_v2.action_space.low)
        observations_v2.append(obs)
        rewards_v2.append(reward)
    env_v2.close()

    env_v3 = World(task=task,
                   enable_visualization=False,
                   seed=54)

    obs = env_v3.reset()
    observations_v3.append(obs)
    for _ in range(horizon):
        obs, reward, done, info = env_v3.step(env_v3.action_space.low)
        observations_v3.append(obs)
        rewards_v3.append(reward)
    env_v3.close()

    assert all(not np.array_equal(observations_v1[i], observations_v3[i])
               for i in range(horizon))
    assert not rewards_v1 == rewards_v3
