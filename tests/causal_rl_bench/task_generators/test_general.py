from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
import numpy as np
import unittest


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.task = task_generator(task_generator_id="general")
        self.env = World(task=self.task,
                         enable_visualization=False)
        return

    def tearDown(self):
        self.env.close()
        return

    def test_determinism(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env._action_space.sample() for _ in range(horizon)]
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
                if not np.array_equal(observations_1[i], observations_2[i]):
                    print(np.array(observations_1[i]) - np.array(observations_2[i]))
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def test_determinism_w_interventions(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env._action_space.sample() for _ in range(horizon)]
        actions = np.array(actions)
        new_goal = self.env.sample_new_goal()
        self.env.do_intervention(new_goal)
        self.env.reset()
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            observations_1.append(obs)
            rewards_1.append(reward)

        for _ in range(10):
            observations_2 = []
            rewards_2 = []
            self.env.reset()
            for i in range(horizon):
                obs, reward, done, info = self.env.step(actions[i])
                observations_2.append(obs)
                rewards_2.append(reward)
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def test_determinism_w_in_episode_interventions(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env._action_space.sample() for _ in range(horizon)]
        actions = np.array(actions)
        self.env.reset()
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            observations_1.append(obs)
            rewards_1.append(reward)
        #now we will restart again and perform an in epsiode intervention
        self.env.reset()
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            if i == 50:
                success_signal = self.env.do_intervention({'tool_level_0_num_1': {'position': [0, 0, 2]}})
        observations_2 = []
        rewards_2 = []
        self.env.reset()
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            observations_2.append(obs)
            rewards_2.append(reward)
            assert np.array_equal(observations_1[i], observations_2[i])
        assert rewards_1 == rewards_2
