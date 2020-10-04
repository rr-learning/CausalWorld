from causal_world.task_generators.task import generate_task
from causal_world.utils.rotation_utils import cyl2cart
from causal_world.envs.causalworld import CausalWorld
import numpy as np
import unittest


class TestPickAndPlace(unittest.TestCase):

    def setUp(self):
        self.task = generate_task(task_generator_id="pick_and_place")
        self.env = CausalWorld(task=self.task, enable_visualization=False)
        return

    def tearDown(self):
        self.env.close()
        return

    def test_determinism(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
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
                if not np.array_equal(observations_1[i], observations_2[i]):
                    print(
                        np.array(observations_1[i]) -
                        np.array(observations_2[i]))
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def test_determinism_w_interventions(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env.action_space.sample() for _ in range(horizon)]
        actions = np.array(actions)
        new_goal = self.env.sample_new_goal()
        self.env.set_starting_state(interventions_dict=new_goal)
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
                if not np.array_equal(observations_1[i], observations_2[i]):
                    print(i)
                    print(
                        np.array(observations_1[i]) -
                        np.array(observations_2[i]))
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def test_determinism_w_in_episode_interventions(self):
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env.action_space.sample() for _ in range(horizon)]
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
                success_signal = self.env.do_intervention(
                    {'tool_block': {
                        'cylindrical_position': [0.1, np.pi / 2, 0.0325]
                    }})
        observations_2 = []
        rewards_2 = []
        self.env.reset()
        for i in range(horizon):
            obs, reward, done, info = self.env.step(actions[i])
            observations_2.append(obs)
            rewards_2.append(reward)
            assert np.array_equal(observations_1[i], observations_2[i])
        assert rewards_1 == rewards_2

    def test_goal_intervention(self):
        task = generate_task(task_generator_id='pick_and_place')
        env = CausalWorld(task=task,
                          enable_visualization=False,
                          normalize_observations=False)
        for _ in range(10):
            invalid_interventions_before = env.get_tracker(
            ).invalid_intervention_steps
            new_goal = env.sample_new_goal()
            env.set_starting_state(interventions_dict=new_goal)
            invalid_interventions_after = env.get_tracker(
            ).invalid_intervention_steps
            for _ in range(2):
                for _ in range(100):
                    obs, reward, done, info = env.step(env.action_space.low)
                    #TODO: this shouldnt be the case when the benchmark is complete
                    #Its a hack for now
                    if invalid_interventions_before == invalid_interventions_after:
                        assert np.array_equal(
                            cyl2cart(
                                new_goal['goal_block']['cylindrical_position']),
                            obs[-18:-15])
                env.reset()

        env.close()