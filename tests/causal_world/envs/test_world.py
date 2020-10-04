from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
import numpy as np
import unittest


class TestWorld(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        return

    def test_determinism(self):
        task = generate_task(task_generator_id="stacked_blocks")
        observations_v1 = []
        observations_v2 = []
        observations_v3 = []
        rewards_v1 = []
        rewards_v2 = []
        rewards_v3 = []
        horizon = 30

        env_v1 = CausalWorld(task=task, enable_visualization=False, seed=27)

        obs = env_v1.reset()
        observations_v1.append(obs)
        for _ in range(horizon):
            obs, reward, done, info = env_v1.step(env_v1.action_space.low)
            observations_v1.append(obs)
            rewards_v1.append(reward)
        env_v1.close()

        task = generate_task(task_generator_id="stacked_blocks")
        env_v2 = CausalWorld(task=task, enable_visualization=False, seed=27)

        obs = env_v2.reset()
        observations_v2.append(obs)
        for _ in range(horizon):
            obs, reward, done, info = env_v2.step(env_v2.action_space.low)
            observations_v2.append(obs)
            rewards_v2.append(reward)
        env_v2.close()

        task = generate_task(task_generator_id="stacked_blocks")
        env_v3 = CausalWorld(task=task, enable_visualization=False, seed=54)

        obs = env_v3.reset()
        observations_v3.append(obs)
        for _ in range(horizon):
            obs, reward, done, info = env_v3.step(env_v3.action_space.low)
            observations_v3.append(obs)
            rewards_v3.append(reward)
        env_v3.close()

        assert all(
            np.array_equal(observations_v1[i], observations_v2[i])
            for i in range(horizon))
        assert rewards_v1 == rewards_v2
        assert all(
            np.array_equal(observations_v1[i], observations_v3[i])
            for i in range(horizon))
        assert rewards_v1 == rewards_v3

    def test_parallelism(self):
        task = generate_task(task_generator_id="stacked_blocks")
        env1 = CausalWorld(task=task, enable_visualization=False, seed=0)
        env1.reset()
        task2 = generate_task(task_generator_id="stacked_blocks")
        env2 = CausalWorld(task=task2, enable_visualization=False, seed=0)
        observations_env1_v1, rewards_env1_v1, _, _ = env1.step(
            env1.action_space.low)
        env2.reset()
        observations_env2_v1, rewards_env2_v1, _, _ = env2.step(
            env2.action_space.low)
        env1.close()
        env2.close()
        assert np.array_equal(observations_env2_v1, observations_env1_v1)
        return

    def test_timing_profile(self):
        from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
        import time

        kuka_env = KukaGymEnv(renders=False,
                              isDiscrete=False)  # operates at 240 HZ
        task = generate_task(task_generator_id="pushing")
        causal_rl_env = CausalWorld(
            task=task,
            enable_visualization=False,
            seed=0,
            skip_frame=10,
            normalize_actions=False,
            normalize_observations=False)  # operates at 250 HZ
        start = time.time()
        kuka_env.reset()
        end = time.time()
        kuka_reset_time = end - start

        start = time.time()
        causal_rl_env.reset()
        end = time.time()
        causal_rl_reset_time = end - start

        self.assertLess(causal_rl_reset_time, kuka_reset_time * 1.25)

        start = time.time()
        kuka_env.step(kuka_env.action_space.sample())
        end = time.time()
        kuka_step_time = end - start

        start = time.time()
        causal_rl_env.step(causal_rl_env.action_space.sample())
        end = time.time()
        causal_rl_step_time = end - start
        print("time 1", causal_rl_step_time)
        print("time 2", kuka_step_time)
        self.assertLess(causal_rl_step_time, kuka_step_time * 10)

        start = time.time()
        kuka_env.render()
        end = time.time()
        kuka_render_time = end - start

        start = time.time()
        causal_rl_env.render()
        end = time.time()
        causal_rl_render_time = end - start
        self.assertLess(causal_rl_render_time, kuka_render_time * 1.25)

        causal_rl_env.close()
        kuka_env.close()
        return

    def test_camera_timing_profile(self):
        from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
        import time

        kuka_env = KukaCamGymEnv(renders=False,
                                 isDiscrete=False)  # operates at 240 HZ
        task = generate_task(task_generator_id="pushing")
        causal_rl_env = CausalWorld(
            task=task,
            enable_visualization=False,
            seed=0,
            skip_frame=10,
            normalize_actions=False,
            normalize_observations=False)  # operates at 250 HZ
        start = time.time()
        kuka_env.reset()
        end = time.time()
        kuka_reset_time = end - start

        start = time.time()
        causal_rl_env.reset()
        end = time.time()
        causal_rl_reset_time = end - start
        self.assertLess(causal_rl_reset_time, kuka_reset_time * 3)

        start = time.time()
        kuka_env.step(kuka_env.action_space.sample())
        end = time.time()
        kuka_step_time = end - start

        start = time.time()
        causal_rl_env.step(causal_rl_env.action_space.sample())
        end = time.time()
        causal_rl_step_time = end - start
        self.assertLess(causal_rl_step_time, kuka_step_time * 1)

        causal_rl_env.close()
        kuka_env.close()
        return

    @unittest.skip("This test will fail because of a potential bug in pybullet")
    def test_save_state(self):
        task = generate_task(task_generator_id="creative_stacked_blocks")
        env = CausalWorld(task=task, enable_visualization=False, seed=0)
        actions = [env.action_space.sample() for _ in range(200)]
        env.reset()
        observations_1 = []
        rewards_1 = []
        for i in range(200):
            observations, rewards, _, _ = env.step(actions[i])
            if i == 100:
                state = env.get_state()
            observations_1.append(observations)
            rewards_1.append(rewards)
        env.set_state(state)
        for i in range(101, 200):
            observations, rewards, _, _ = env.step(actions[i])
            if not np.array_equal(observations_1[i], observations):
                print("step", i)
                print(observations_1[i] - observations)
            assert np.array_equal(observations_1[i], observations)
        env.close()
        return

    def test_reset_default_state(self):
        task = generate_task(task_generator_id="picking")
        env = CausalWorld(task=task, enable_visualization=False, seed=0)
        actions = [env.action_space.sample() for _ in range(200)]
        observations_1 = []
        rewards_1 = []
        env.reset()
        for i in range(200):
            observations, rewards, _, _ = env.step(actions[i])
            observations_1.append(observations)
            rewards_1.append(rewards)
        env.set_starting_state(
            {'goal_block': {
                'cylindrical_position': [0.1, np.pi, 0.1]
            }})
        for i in range(200):
            observations, rewards, _, _ = env.step(actions[i])
        env.reset_default_state()
        for i in range(200):
            observations, rewards, _, _ = env.step(actions[i])
            assert np.array_equal(observations_1[i], observations)
        env.close()
        return


if __name__ == '__main__':
    unittest.main()
