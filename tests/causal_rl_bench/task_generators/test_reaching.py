from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
import numpy as np
import unittest


class TestReaching(unittest.TestCase):
    def setUp(self):
        self.task = task_generator(task_generator_id="reaching")
        self.env = World(task=self.task,
                         enable_visualization=False)
        return

    def tearDown(self):
        self.env.close()
        return

    def test_determinism(self):
        observations_1 = []
        rewards_1 = []
        horizon = 2000
        actions = [self.env.action_space.sample() for _ in range(horizon)]
        actions = np.array(actions)
        obs = self.env.reset()
        observations_1.append(obs)
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            observations_1.append(obs)
            rewards_1.append(reward)

        for _ in range(10):
            observations_2 = []
            rewards_2 = []
            obs = self.env.reset()
            observations_2.append(obs)
            for i in range(horizon):
                obs, reward, done, info = self.env.step(actions[i])
                observations_2.append(obs)
                rewards_2.append(reward)
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2
