from causal_rl_bench.utils.task_utils import get_suggested_grip_locations
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
import numpy as np
import unittest


class TestPicking(unittest.TestCase):
    def setUp(self):
        self.task = task_generator(task_generator_id="picking")
        self.env = World(task=self.task,
                         enable_visualization=False,
                         skip_frame=1,
                         action_mode="end_effector_positions",
                         normalize_actions=False,
                         normalize_observations=False)
        return

    def tearDown(self):
        self.env.close()
        return

    def test_determinism(self):
        self.env.set_action_mode('joint_positions')
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

    def lift_last_finger_first(self, current_obs):
        desired_action = current_obs[27:27 + 9]
        desired_action[6:] = [-0, -0.08, 0.4]
        for _ in range(250):
            obs, reward, done, info = self.env.step(desired_action)
        return desired_action

    def grip_block(self):
        grip_locations = get_suggested_grip_locations(self.env._task.stage.get_object('tool_block').get_size(),
                                                      self.env._task.stage.get_object('tool_block').world_to_cube_r_matrix())
        desired_action = np.zeros(9)
        desired_action[6:] = [-0, -0.08, 0.4]
        desired_action[:3] = grip_locations[0]
        desired_action[3:6] = grip_locations[1]
        # grasp the block now
        for _ in range(250):
            obs, reward, done, info = self.env.step(desired_action)
        return desired_action

    def lift_block(self, desired_grip):
        desired_action = desired_grip
        for _ in range(50):
            desired_action[2] += 0.005
            desired_action[5] += 0.005
            for _ in range(10):
                obs, reward, done, info = self.env.step(desired_action)
        return obs

    # def test_02_mass(self):
    #     self.env.set_action_mode('end_effector_positions')
    #     intervention = {'tool_block': {'mass': 0.02}}
    #     self.env.do_intervention(interventions_dict=intervention)
    #     for _ in range(1):
    #         obs = self.env.reset()
    #         self.lift_last_finger_first(obs)
    #         desired_grip = self.grip_block()
    #         self.assertEqual(self.env.get_robot().get_tip_contact_states(), [1, 1, 0], "contact states are not closed")
    #         final_obs = self.lift_block(desired_grip)
    #         self.assertGreater(final_obs[38], 0.2, "the block didn't get lifted")
    #
    # def test_08_mass(self):
    #     self.env.set_action_mode('end_effector_positions')
    #     intervention = {'tool_block': {'mass': 0.08}}
    #     self.env.do_intervention(interventions_dict=intervention)
    #     for _ in range(1):
    #         obs = self.env.reset()
    #         self.lift_last_finger_first(obs)
    #         desired_grip = self.grip_block()
    #         self.assertEqual(self.env.get_robot().get_tip_contact_states(), [1, 1, 0], "contact states are not closed")
    #         final_obs = self.lift_block(desired_grip)
    #         # self.assertGreater(final_obs[38], 0.2, "the block didn't get lifted")
    #
    # def test_1_mass(self):
    #     self.env.set_action_mode('end_effector_positions')
    #     intervention = {'tool_block': {'mass': 0.1}}
    #     self.env.do_intervention(interventions_dict=intervention)
    #     for _ in range(1):
    #         obs = self.env.reset()
    #         self.lift_last_finger_first(obs)
    #         desired_grip = self.grip_block()
    #         self.assertEqual(self.env.get_robot().get_tip_contact_states(), [1, 1, 0], "contact states are not closed")
    #         final_obs = self.lift_block(desired_grip)
    #         # self.assertGreater(final_obs[38], 0.2, "the block didn't get lifted")

    def test_determinism_w_interventions(self):
        self.env.set_action_mode('joint_positions')
        observations_1 = []
        rewards_1 = []
        horizon = 100
        actions = [self.env.action_space.sample() for _ in range(horizon)]
        actions = np.array(actions)
        new_goal = self.env.sample_new_goal()
        self.env.reset(interventions_dict=new_goal)
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
                print("current step ", i)
                print(np.array(observations_1[i]) -
                      np.array(observations_2[i]))
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def test_determinism_w_in_episode_interventions(self):
        self.env.set_action_mode('joint_positions')
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
                success_signal = self.env.do_intervention({'tool_block':
                                                               {'position':
                                                                    [0.1, 0.1,
                                                                     0.0425]}})
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
        task = task_generator(task_generator_id='picking')
        env = World(task=task, enable_visualization=False, normalize_observations=False)
        for _ in range(10):
            invalid_interventions_before = env.get_tracker().invalid_intervention_steps
            new_goal = env.sample_new_goal()
            env.reset(interventions_dict=new_goal)
            invalid_interventions_after = env.get_tracker().invalid_intervention_steps
            for _ in range(2):
                for _ in range(100):
                    obs, reward, done, info = env.step(env.action_space.low)
                    #TODO: this shouldnt be the case when the benchmark is complete
                    #Its a hack for now
                    if invalid_interventions_before == invalid_interventions_after:
                        assert np.array_equal(new_goal['goal_block']['position'],
                                              obs[-7:-4])
                env.reset()

        env.close()

