# import robot_fingers
# import robot_interfaces
#
#
# def perform_step_real_robot(frontend, position, repetitions):
#     for i in range(repetitions):
#         t = frontend.append_desired_action(
#             robot_interfaces.trifinger.Action(position=position))
#         frontend.wait_until_time_index(t)
#     current_position = frontend.get_observation(t).position
#     current_velocity = frontend.get_observation(t).velocity
#     return current_position
#
#
# def perform_step_simulated_robot(env, position, repetitions):
#     for _ in range(repetitions):
#         obs, reward, done, info = env.step(position)
#     return obs[:9]
#
#
# def test_pd_gains():
#     # control the robot using pd controller
#     from causal_world.envs.world import World
#     from causal_world.tasks.task import Task
#     import numpy as np
#     np.random.seed(0)
#     task = Task(task_generator_id='reaching')
#     skip_frame = 1 #@250Hz
#     threshold = 0.05
#     env = World(task=task, enable_visualization=True, skip_frame=skip_frame, normalize_observations=False,
#                 normalize_actions=False, seed=0)
#     robot = robot_fingers.Robot(robot_interfaces.trifinger,
#                                 robot_fingers.create_trifinger_backend,
#                                 "trifinger.yml")
#     robot.initialize()
#     frontend = robot.frontend
#     zero_hold_real_robot = 1000
#     zero_hold_simulator = 250
#     obs = env.reset()
#     #stay at the idle place for the zero hold
#     desired_action = obs[:9]
#     simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#     real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#     if (np.abs(real_positions - simulated_positions) > threshold).any():
#         raise AssertionError("staying at idle position failed")
#
#     # checking upper bound limit
#     #TODO: this fails! the stage position is not accurate for now
#     # desired_action = env.action_space.high
#     # simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator * 10)
#     # real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot * 5)
#     # if (np.abs(real_positions - simulated_positions) > threshold).any():
#     #     raise AssertionError("going to upper bound failed")
#
#     #checking lower bound limit
#     desired_action = env.action_space.low
#     simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#     real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#     if (np.abs(real_positions - simulated_positions) > threshold).any():
#         raise AssertionError("going to lower bound failed")
#
#     #test each finger by itself
#     for i in range(3):
#         #set all the fingers to low bound first
#         default_desired_action = env.action_space.low
#         simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#         real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#         for j in range(100):
#             desired_action = default_desired_action
#             desired_action[i*3:(i+1)*3] = env.__robot.sample_joint_positions()[i * 3:(i + 1) * 3]
#             simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#             real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#             if (np.abs(real_positions - simulated_positions) > threshold).any():
#                 raise AssertionError("random position failed comparison")
#
#     # check random positions now with possible collisions
#     for i in range(100):
#         desired_action = env.__robot.sample_joint_positions()
#         simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#         real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#         if (np.abs(real_positions - simulated_positions) > threshold).any():
#             raise AssertionError("random position failed comparison")
#
#     env.close()
#
#
# def test_pd_gains_2():
#     # control the robot using pd controller
#     from causal_world.envs.world import World
#     from causal_world.tasks.task import Task
#     import numpy as np
#     np.random.seed(0)
#     task = Task(task_generator_id='reaching')
#     skip_frame = 1#@240Hz
#     threshold = 0.05
#     env = World(task=task, enable_visualization=True, skip_frame=skip_frame, normalize_observations=False,
#                 normalize_actions=False, seed=0)
#     robot = robot_fingers.Robot(robot_interfaces.trifinger,
#                                 robot_fingers.create_trifinger_backend,
#                                 "trifinger.yml")
#     robot.initialize()
#     frontend = robot.frontend
#     zero_hold_real_robot = 4
#     zero_hold_simulator = 1
#     obs = env.reset()
#     #stay at the idle place for the zero hold
#     desired_action = env.action_space.low
#     # simulated_positions = perform_step_simulated_robot(env, desired_action_sim, zero_hold_simulator)
#     # real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#     # if (np.abs(real_positions - simulated_positions) > threshold).any():
#     #     raise AssertionError("staying at idle position failed")
#     #
#     # desired_action = env.action_space.low
#     # simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#     # real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#     # if (np.abs(real_positions - simulated_positions) > threshold).any():
#     #     raise AssertionError("going to lower bound failed")
#     for i in range(250):
#         simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#         real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#     #go to low space in both envs
#     for _ in range(5000):
#         simulated_positions = perform_step_simulated_robot(env, simulated_positions, zero_hold_simulator)
#         real_positions = perform_step_real_robot(frontend, real_positions, zero_hold_real_robot)
#             # desired_action = np.zeros([9,])
#             # current_obs = np.around(obs[:9], decimals=2)
#             # print("what I wanted to reach", current_obs + desired_action)
#             # obs, reward, done, info = env.step(desired_action)
#             # print("what I actually reached", np.around(obs[:9], decimals=2))
#             # print("diff is", current_obs + desired_action - np.around(obs[:9], decimals=2))
#             #     # desired_action = obs[:9]
#
#         # for j in range(100):
#         #     desired_action = default_desired_action
#         #     desired_action[i * 3:(i + 1) * 3] = env.robot.sample_joint_positions()[i * 3:(i + 1) * 3]
#         #     simulated_positions = perform_step_simulated_robot(env, desired_action, zero_hold_simulator)
#         #     real_positions = perform_step_real_robot(frontend, desired_action, zero_hold_real_robot)
#         #     if (np.abs(real_positions - simulated_positions) > threshold).any():
#         #         raise AssertionError("random position failed comparison")
#
#
# # test_pd_gains()
# # test_pd_gains_2()
