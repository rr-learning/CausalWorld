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

    assert all(np.array_equal(observations_v1[i], observations_v2[i])
               for i in range(horizon))
    assert rewards_v1 == rewards_v2

    assert all(not np.array_equal(observations_v1[i], observations_v3[i])
               for i in range(horizon))
    assert not rewards_v1 == rewards_v3


def test_parallelism():
    task = Task(task_id="pushing")
    env1 = World(task=task,
                 enable_visualization=False,
                 seed=0)
    env1.reset()
    task2 = Task(task_id="pushing")
    env2 = World(task=task2,
                 enable_visualization=False,
                 seed=0)
    observations_env1_v1, rewards_env1_v1, _, _ = env1.step(env1.action_space.low)
    env2.reset()
    observations_env2_v1, rewards_env2_v1, _, _ = env2.step(env2.action_space.low)
    env1.close()
    env2.close()
    assert np.array_equal(observations_env2_v1,  observations_env1_v1)
    return


def timing_profile():
    from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
    import time

    kuka_env = KukaGymEnv(renders=False, isDiscrete=False)#operates at 240 HZ
    task = Task(task_id="pushing")
    causal_rl_env = World(task=task,
                          enable_visualization=False,
                          seed=0,
                          skip_frame=1,
                          normalize_actions=False,
                          normalize_observations=False)#operates at 250 HZ
    start = time.time()
    kuka_env.reset()
    end = time.time()
    print("kuka reset:", end-start)

    start = time.time()
    causal_rl_env.reset()
    end = time.time()
    print("causal_rl reset:", end - start)

    start = time.time()
    kuka_env.step(kuka_env.action_space.sample())
    end = time.time()
    print("kuka step:", end - start)

    start = time.time()
    causal_rl_env.step(causal_rl_env.action_space.sample())
    end = time.time()
    print("causal_rl step:", end - start)

    start = time.time()
    kuka_env.render()
    end = time.time()
    print("kuka render:", end - start)

    start = time.time()
    causal_rl_env.render()
    end = time.time()
    print("causal_rl render:", end - start)
