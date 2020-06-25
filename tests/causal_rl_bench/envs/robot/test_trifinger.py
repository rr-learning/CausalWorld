from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
import numpy as np
import pytest


@pytest.fixture(scope="module")
def robot_jp_structured():
    task = task_generator(task_generator_id='pushing')
    return World(task=task, enable_visualization=False, observation_mode="structured")


@pytest.fixture(scope="module")
def robot_jp_camera():
    task = task_generator(task_generator_id='pushing')
    return World(task=task, enable_visualization=False, observation_mode="cameras")


def test_action_mode_switching(robot_jp_structured):
    robot_jp_structured.set_action_mode("joint_torques")
    assert robot_jp_structured.get_action_mode() == "joint_torques"
    robot_jp_structured.set_action_mode("joint_positions")
    assert robot_jp_structured.get_action_mode() == "joint_positions"


@pytest.mark.skip
def test_turn_on_cameras():
    assert False


@pytest.mark.skip
def test_turn_off_cameras():
    assert False


@pytest.mark.skip
def test_apply_action():
    assert False


@pytest.mark.skip
def test_get_full_state():
    assert False


@pytest.mark.skip
def test_set_full_state():
    assert False


@pytest.mark.skip
def test_reset_robot_state():
    assert False


@pytest.mark.skip
def test_get_last_action_applied():
    assert False


@pytest.mark.skip
def test_get_current_full_observations():
    assert False


@pytest.mark.skip
def test_get_current_partial_observations():
    assert False


@pytest.mark.skip
def test_get_tip_positions():
    assert False


@pytest.mark.skip
def test_get_observation_space():
    assert False


@pytest.mark.skip
def test_get_action_spaces():
    assert False


def test_pd_gains():
    #control the robot using pd controller
    np.random.seed(0)
    task = task_generator(task_generator_id='pushing')
    skip_frame = 1
    env = World(task=task, enable_visualization=False, skip_frame=skip_frame, normalize_observations=False,
                normalize_actions=False, seed=0)
    zero_hold = int(5000 / skip_frame) #reach desired position in 4 secs?
    obs = env.reset()
    #test bounds first

    for _ in range(zero_hold):
        chosen_action = env.action_space.high
        obs, reward, done, info = env.step(chosen_action)
    current_joint_positions = obs[:9]
    if (((current_joint_positions - chosen_action) > 0.1).any()):
        raise AssertionError("The pd controller failed to reach these values {} but reached instead {}".
                             format(chosen_action, current_joint_positions))

    for _ in range(zero_hold):
        chosen_action = env.action_space.low
        obs, reward, done, info = env.step(chosen_action)
    current_joint_positions = obs[:9]
    if (((current_joint_positions - chosen_action) > 0.1).any()):
        raise AssertionError("The pd controller failed to reach these values {} but reached instead {}".
                             format(chosen_action, current_joint_positions))

    # for i in range(200):
    #     #check for first finger
    #     chosen_action = np.random.uniform(env.action_space.low, env.action_space.high)
    #     chosen_action[3:] = env.action_space.low[3:]
    #     chosen_action[1] = 0
    #     chosen_action[2] = 0
    #     for _ in range(zero_hold):
    #         chosen_action = chosen_action
    #         obs, reward, done, info = env.step(chosen_action)
    #     current_joint_positions = obs[:9]
    #     if(((current_joint_positions - chosen_action) > 0.1).any()):
    #         raise AssertionError("The pd controller failed to reach these values {} but reached instead {}".
    #                              format(chosen_action, current_joint_positions))
    env.close()
