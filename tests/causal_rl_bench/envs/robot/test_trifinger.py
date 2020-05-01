from causal_rl_bench.envs.robot.trifinger import TriFingerRobot

import pytest


@pytest.fixture(scope="module")
def robot_jp_structured():
    return TriFingerRobot(action_mode="joint_positions", observation_mode="structured", enable_visualization=False)


@pytest.fixture(scope="module")
def robot_jp_camera():
    return TriFingerRobot(action_mode="joint_positions", observation_mode="cameras", enable_visualization=False)


def test_action_mode_switching(robot_jp_structured):
    robot_jp_structured.set_action_mode("joint_torques")
    assert robot_jp_structured.get_action_mode() == "joint_torques"
    robot_jp_structured.set_action_mode("joint_positions")
    assert robot_jp_structured.get_action_mode() == "joint_positions"


def test_observation_mode_switch(robot_jp_structured):
    robot_jp_structured.set_observation_mode("cameras")
    assert robot_jp_structured.get_observation_mode() == "cameras"
    robot_jp_structured.set_observation_mode("structured")
    assert robot_jp_structured.get_observation_mode() == "structured"


def test_camera_skip_frame(robot_jp_structured):
    assert robot_jp_structured.get_camera_skip_frame() == 0.3
    robot_jp_structured.set_camera_skip_frame(0.8)
    assert robot_jp_structured.get_camera_skip_frame() == 0.8
    robot_jp_structured.set_camera_skip_frame(0.3)


def test_skip_frame(robot_jp_structured):
    assert robot_jp_structured.get_skip_frame() == 0.02
    robot_jp_structured.set_skip_frame(0.03)
    assert robot_jp_structured.get_skip_frame() == 0.03
    robot_jp_structured.set_skip_frame(0.02)


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
