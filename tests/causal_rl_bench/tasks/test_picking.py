from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
import numpy as np
from causal_rl_bench.utils.task_utils import get_suggested_grip_locations


def lift_last_finger_first(env, current_obs):
    desired_action = current_obs[27:27+9]
    desired_action[6:] = [-0, -0.08, 0.4]
    for _ in range(250):
        obs, reward, done, info = env.step(desired_action)
    return desired_action


def grip_block(env):
    grip_locations = get_suggested_grip_locations(env.task.stage.get_object('block').size,
                                                  env.task.stage.get_object('block').world_to_cube_r_matrix())
    desired_action = np.zeros(9)
    desired_action[6:] = [-0, -0.08, 0.4]
    desired_action[:3] = grip_locations[0]
    desired_action[3:6] = grip_locations[1]
    print("wants to reach", desired_action)
    # grasp the block now
    for _ in range(250):
        obs, reward, done, info = env.step(desired_action)
    print("reached instead ", obs[27:27+9])
    return desired_action


def lift_block(env, desired_grip):
    desired_action = desired_grip
    for _ in range(100):
        desired_action[2] += 0.005
        desired_action[5] += 0.005
        for _ in range(10):
            obs, reward, done, info = env.step(desired_action)


def test_mass():
    task = Task(task_id='picking', randomize_joint_positions=False,
                randomize_block_pose=False, block_mass=0.1)
    env = World(task=task, skip_frame=1, enable_visualization=True, seed=0,
                action_mode="end_effector_positions",
                observation_mode="structured",
                normalize_actions=False,
                normalize_observations=False,
                max_episode_length=10000)
    for _ in range(10):
        obs = env.reset()
        lift_last_finger_first(env, obs)
        desired_grip = grip_block(env)
        print(env.robot.get_tip_contact_states())
        lift_block(env, desired_grip)
    env.close()


test_mass()



