from causal_world.utils.task_utils import get_suggested_grip_locations
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.utils.rotation_utils import cyl2cart
import numpy as np
import unittest


class TestPicking(unittest.TestCase):

    def setUp(self):
        self.task = generate_task(task_generator_id="picking")
        self.env = CausalWorld(task=self.task,
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
                if not np.array_equal(observations_1[i], observations_2[i]):
                    print(observations_1[i] - observations_2[i])
                assert np.array_equal(observations_1[i], observations_2[i])
            assert rewards_1 == rewards_2

    def lift_last_finger_first(self, current_obs):
        desired_action = current_obs[19:19 + 9]
        desired_action[6:] = [-0, -0.08, 0.4]
        for _ in range(250):
            obs, reward, done, info = self.env.step(desired_action)
        return desired_action

    def move_first_two_fingers(self, current_obs):
        desired_action = current_obs[19:19 + 9]
        desired_action[:6] = [0., 0.15313708, 0.05586292, 0.13262061, -0.07656854, 0.05586292]
        for _ in range(250):
            obs, reward, done, info = self.env.step(desired_action)
        return obs

    def grip_block(self):
        grip_locations = get_suggested_grip_locations(
            self.env._task._stage.get_object('tool_block').get_size(),
            self.env._task._stage.get_object(
                'tool_block').world_to_cube_r_matrix())
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
        for _ in range(40):
            desired_action[2] += 0.005
            desired_action[5] += 0.005
            for _ in range(10):
                obs, reward, done, info = self.env.step(desired_action)
        return obs

    def test_02_mass(self):
        self.env.set_action_mode('end_effector_positions')
        intervention = {'tool_block': {'mass': 0.02}}
        self.env.do_intervention(interventions_dict=intervention)
        for _ in range(1):
            obs = self.env.reset()
            obs = self.move_first_two_fingers(obs)
            self.lift_last_finger_first(obs)
            desired_grip = self.grip_block()
            self.assertEqual(self.env.get_robot().get_tip_contact_states(),
                             [1, 1, 0], "contact states are not closed")
            final_obs = self.lift_block(desired_grip)
            self.assertGreater(final_obs[-22], 0.2,
                               "the block didn't get lifted")

    def test_08_mass(self):
        self.env.set_action_mode('end_effector_positions')
        intervention = {'tool_block': {'mass': 0.08}}
        self.env.do_intervention(interventions_dict=intervention)
        for _ in range(1):
            obs = self.env.reset()
            obs = self.move_first_two_fingers(obs)
            self.lift_last_finger_first(obs)
            desired_grip = self.grip_block()
            self.assertEqual(self.env.get_robot().get_tip_contact_states(),
                             [1, 1, 0], "contact states are not closed")
            final_obs = self.lift_block(desired_grip)
            self.assertGreater(final_obs[-22], 0.2,
                               "the block didn't get lifted")

    def test_1_mass(self):
        self.env.set_action_mode('end_effector_positions')
        intervention = {'tool_block': {'mass': 0.1}}
        self.env.do_intervention(interventions_dict=intervention)
        for _ in range(1):
            obs = self.env.reset()
            obs = self.move_first_two_fingers(obs)
            self.lift_last_finger_first(obs)
            desired_grip = self.grip_block()
            self.assertEqual(self.env.get_robot().get_tip_contact_states(),
                             [1, 1, 0], "contact states are not closed")
            final_obs = self.lift_block(desired_grip)
            self.assertGreater(final_obs[-22], 0.2,
                               "the block didn't get lifted")

    def test_determinism_w_interventions(self):
        self.env.set_action_mode('joint_positions')
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
            if not np.array_equal(observations_1[i], observations_2[i]):
                print(observations_1[i] - observations_2[i])
            assert np.array_equal(observations_1[i], observations_2[i])
        assert rewards_1 == rewards_2

    def test_goal_intervention(self):
        task = generate_task(task_generator_id='picking')
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
                            obs[-7:-4])
                env.reset()

        env.close()