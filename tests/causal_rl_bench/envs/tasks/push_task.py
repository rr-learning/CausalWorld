from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.state_utils import euler_to_quaternion
import numpy as np
from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
import math
from causal_rl_bench.utils.task_utils import calculate_end_effector_to_goal


class PushingTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.id = "pushing"
        self.robot = None
        self.stage = None
        self.seed = 0

        self.task_solved = False

        #robot observations first
        self.observation_keys = ["joint_positions",
                                 "joint_velocities",
                                 # "action_joint_positions",
                                 # "goal_block_position",
                                 "block_position"]

        self.selected_robot_observations = ["joint_positions",
                                            "joint_velocities"]

    def init_task(self, robot, stage):
        self.robot = robot
        self.stage = stage
        self.stage.add_rigid_general_object(name="block",
                                            shape="cube", mass=0.005)
        self.stage.add_silhoutte_general_object(name="goal_block",
                                                shape="cube")
        # self.robot.add_observation(observation_key="end_effector_to_goal",
        #                            low_bound=[-0.5] * 3 * 3,
        #                            upper_bound=[0.5] * 3 * 3)
        self.robot.add_observation(observation_key="action_joint_positions",
                                   low_bound=np.array([-math.radians(70), -math.radians(70),
                                                       -math.radians(160)] * 3),
                                   upper_bound=np.array([math.radians(70), math.radians(0),
                                                         math.radians(-2)] * 3))

        self.stage.finalize_stage()
        # self.stage.add_observation(observation_key="action_joint_positions",
        #                            low_bound=np.array([-math.radians(90), -math.radians(90),
        #                                                -math.radians(172)] * 3),
        #                            upper_bound=np.array([math.radians(90), math.radians(100),
        #                                                  math.radians(-2)] * 3))
        return

    def reset_task(self):
        sampled_positions = self.robot.sample_positions()
        self.robot.clear()
        self.stage.clear()
        self.robot.set_full_state(np.append(sampled_positions,
                                            np.zeros(9)))
        self.task_solved = False
        self.reset_scene_objects()
        task_observations = self.filter_observations()
        return task_observations

    def get_description(self):
        return \
            "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        block_state = self.stage.get_object_state('block')
        # robot_observations = self.robot.get_current_full_observations()
        block_position = block_state["block_position"]
        # end_effector_positions = robot_observations["end_effector_positions"].reshape(-1, 3)
        # distance_from_block = np.sum(
        #     (end_effector_positions - block_position) ** 2)
        # reward = - 1.3 * distance_from_block
        TARGET_HEIGHT = 0.1
        z = block_position[-1]
        x = block_position[0]
        y = block_position[1]
        reward = -abs(z - TARGET_HEIGHT) - (x ** 2 + y ** 2)
        return reward

    def is_done(self):
        return self.task_solved

    def filter_observations(self):
        robot_observations_dict = self.robot.get_current_observations(self.selected_robot_observations)
        stage_observations_dict = self.stage.get_current_observations()
        full_observations_dict = dict(robot_observations_dict)
        full_observations_dict.update(stage_observations_dict)
        observations_filtered = np.array([])
        for key in self.observation_keys:
            if key == "end_effector_to_goal":
                new_obs = calculate_end_effector_to_goal(end_effector_position=full_observations_dict['end_effector_positions'],
                                               goal_position=full_observations_dict['goal_block_position'])
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(new_obs))
            elif key == "action_joint_positions":
                new_obs = self.robot.get_last_clippd_action()
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(new_obs))
            else:
                observations_filtered = \
                    np.append(observations_filtered,
                              np.array(full_observations_dict[key]))
        return observations_filtered

    def get_counterfactual_variant(self):
        pass

    def reset_scene_objects(self):
        # TODO: Refactor the orientation sampling into a general util method

        block_position = self.stage.random_position(height_limits=0.0435)
        block_orientation = euler_to_quaternion([0, 0,
                                                 np.random.uniform(-np.pi,
                                                                   np.pi)])

        goal_position = self.stage.random_position(height_limits=0.0435)
        goal_orientation = euler_to_quaternion([0, 0,
                                                np.random.uniform(-np.pi,
                                                                  np.pi)])
        self.stage.set_objects_pose(names=["block", "goal_block"],
                                    positions=[block_position, goal_position],
                                    orientations=[block_orientation,
                                                  goal_orientation])

    def get_task_params(self):
        task_params_dict = dict()
        task_params_dict["task_id"] = self.id
        task_params_dict["skip_frame"] = self.robot.get_skip_frame()
        task_params_dict["seed"] = self.seed
        task_params_dict["action_mode"] = self.robot.get_action_mode()
        task_params_dict["observation_mode"] = self.robot.get_observation_mode()
        task_params_dict["camera_skip_frame"] = \
            self.robot.get_camera_skip_frame()
        task_params_dict["normalize_actions"] = \
            self.robot.robot_actions.is_normalized()
        task_params_dict["normalize_observations"] = \
            self.robot.robot_observations.is_normalized()
        task_params_dict["max_episode_length"] = None
        return task_params_dict

    def do_random_intervention(self):
        #TODO: for now just intervention on a specific object
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        # self.stage.object_intervention("block", interventions_dict)
        interventions_dict = dict()
        goal_block_position = self.stage.random_position(height_limits=0.0425)
        new_size = np.random.uniform([0.065], [0.15], size=[3,])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("goal_block", interventions_dict)
        return

# import gym
# env1 = gym.make("pybullet_fingers.gym_wrapper:pick-v0", control_rate_s=0.001, seed=0, enable_visualization=True)
#
# for i in range(2000):
#     obs = env1.step(np.ones(9,))
#     # obs, reward, done, info = env1.step(np.zeros(9,))


task = PushingTask()
# #TODO: modify skip frame
env = World(task=task, skip_frame=1, enable_visualization=True, seed=2)
start = time.time()
for i in range(0, 2000):
     obs = env.step(np.ones(9,))
end = time.time()
print(end-start)
    # obs, reward, done, info = env.step(np.zeros(9,))
# current_state = env.get_full_state()
# for i in range(2):
#     # env.reset()
#     # env.set_full_state(current_state)
#     env.do_random_intervention()
#     for j in range(100):
#         recorder.capture_frame()
#         # recorder.capture_frame()
#         # env.step(
#         #     np.random.uniform(env.action_space.low, env.action_space.high,
#         #                       env.action_space.shape))
#         start = time.time()
#         env.step(
#             np.random.uniform(env.action_space.low, env.action_space.high,
#                               env.action_space.shape))
#         end = time.time()
#         print(end - start)
#         # env.render()
#
# recorder.capture_frame()
# recorder.close()
# env.close()