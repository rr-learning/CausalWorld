from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv

kuka_env = KukaCamGymEnv(renders=True,
                         isDiscrete=False)  # operates at 240 HZ

for _ in range(10):
    kuka_env.reset()
    for _ in range(500):
        kuka_env.step(kuka_env.action_space.sample())