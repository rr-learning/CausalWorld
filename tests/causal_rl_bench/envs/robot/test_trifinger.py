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


def test_pd_gains():
    #control the robot using pd controller
    from causal_rl_bench.envs.world import World
    from causal_rl_bench.tasks.task import Task
    import numpy as np
    np.random.seed(0)
    task = Task(task_id='pushing')
    skip_frame = 1
    env = World(task=task, enable_visualization=True, skip_frame=skip_frame, normalize_observations=False,
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
    print("verified upper bound")

    for _ in range(zero_hold):
        chosen_action = env.action_space.low
        obs, reward, done, info = env.step(chosen_action)
    current_joint_positions = obs[:9]
    if (((current_joint_positions - chosen_action) > 0.1).any()):
        raise AssertionError("The pd controller failed to reach these values {} but reached instead {}".
                             format(chosen_action, current_joint_positions))
    print("verified lower bound")
    
    for i in range(200):
        #check for first finger
        chosen_action = np.random.uniform(env.action_space.low, env.action_space.high)
        chosen_action[3:] = env.action_space.low[3:]
        chosen_action[1] = 0
        chosen_action[2] = 0
        for _ in range(zero_hold):
            chosen_action = chosen_action
            obs, reward, done, info = env.step(chosen_action)
        current_joint_positions = obs[:9]
        if(((current_joint_positions - chosen_action) > 0.1).any()):
            raise AssertionError("The pd controller failed to reach these values {} but reached instead {}".
                                 format(chosen_action, current_joint_positions))
    env.close()


def test_inverse_kinemetics():
    from causal_rl_bench.envs.world import World
    from causal_rl_bench.tasks.task import Task
    import numpy as np
    np.random.seed(0)
    skip_frame = 1

    task = Task(task_id='reaching')
    env = World(task=task, enable_visualization=True, skip_frame=skip_frame, normalize_observations=False,
                normalize_actions=False, seed=0, action_mode="end_effector_positions")

    #test 1 stay at the same place
    for _ in range(50):
        obs = env.reset()
        current_end_effector_positions = obs[18:]
        desired_action = np.zeros([9, ]) + current_end_effector_positions
        for _ in range(250):
            obs, reward, done, info = env.step(desired_action)
        if ((np.abs(obs[18:] - desired_action) > 0.01).any()):
            raise AssertionError("The inverse kinemtics failed to reach these values {} but reached instead {}".
                                 format(current_end_effector_positions*100, obs[18:]*100))
    print("Staying at the same place passed")

    #test n random deltas with different episodes
    for i in range(50):
        obs = env.reset()
        current_end_effector_positions = obs[18:]
        for j in range(100):
            #TODO: need to check if its in the feasible set or not
            desired_action = current_end_effector_positions + np.random.uniform(-0.02, 0.02)
            for _ in range(250):
                obs, reward, done, info = env.step(desired_action)
            # if ((np.abs(obs[18:] - desired_action) > 0.03).any()):
            #     raise AssertionError("The inverse kinemtics failed to reach these values {} but reached instead {} "
            #                          "with delta {}".
            #                          format(desired_action, obs[18:], np.abs(obs[18:] - desired_action)))
    env.close()

test_inverse_kinemetics()