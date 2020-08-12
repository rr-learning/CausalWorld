from causal_world.envs.robot.observations import TriFingerObservations

import math
import numpy as np
import pytest


@pytest.fixture(scope='module')
def os_camera_full():
    return TriFingerObservations(observation_mode="pixel",
                                 normalize_observations=False)


@pytest.fixture(scope='module')
def os_structured_full():
    return TriFingerObservations(observation_mode="structured",
                                 observation_keys=[
                                     "action_joint_positions",
                                     "joint_velocities", "joint_torques"
                                 ],
                                 normalize_observations=False)


@pytest.fixture(scope='module')
def os_default():
    return TriFingerObservations()


@pytest.fixture(scope='module')
def os_custom_keys_norm():
    return TriFingerObservations(
        observation_mode="structured",
        normalize_observations=True,
        observation_keys=["end_effector_positions", "action_joint_positions"])


lower_obs_space_structured_full = np.array(
    [-1.57, -1.2, -3.0] * 3 +
    [-50] * 3 * 3 + [-0.36, -0.36, -0.36] * 3)
upper_obs_space_structured_full = np.array(
    [1.0, 1.57, 3.0] * 3 + [50] * 3 * 3 +
    [0.36, 0.36, 0.36] * 3)

upper_99_structured_norm = np.array([0.99] * 27)
upper_100_structured_norm = np.array([1.0] * 27)
upper_101_structured_norm = np.array([1.01] * 27)

lower_99_structured_norm = np.array([-0.99] * 27)
lower_100_structured_norm = np.array([-1.0] * 27)
lower_101_structured_norm = np.array([-1.01] * 27)

lower_obs_space_custom = np.array(
    [-0.5, -0.5, 0.0] * 3 +
    [-1.57, -1.2, -3.0] * 3)
upper_obs_space_custom = np.array(
    [0.5, 0.5, 0.5] * 3 +
    [1.0, 1.57, 3.0] * 3)

normalized_struct_obs_to_be_clipped = np.array([1.02, 0.4, -2] * 9)
normalized_struct_obs_clipped = np.array([1., 0.4, -1.] * 9)


def test_get_observation_spaces(os_default, os_camera_full, os_structured_full,
                                os_custom_keys_norm):
    assert (os_default.get_observation_spaces().low == -1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().low == -1.).all()
    assert (os_default.get_observation_spaces().high == 1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()

    assert (os_structured_full.get_observation_spaces().low ==
            lower_obs_space_structured_full).all()
    assert (os_camera_full.get_observation_spaces().low == 0).all()
    assert (os_structured_full.get_observation_spaces().high ==
            upper_obs_space_structured_full).all()
    assert (os_camera_full.get_observation_spaces().high == 255).all()


def test_is_normalized(os_default, os_camera_full, os_structured_full,
                       os_custom_keys_norm):
    assert os_default.is_normalized()
    assert os_custom_keys_norm.is_normalized()
    assert not os_camera_full.is_normalized()
    assert not os_structured_full.is_normalized()


def test_normalize_observation(os_structured_full, os_custom_keys_norm):
    assert (os_structured_full.normalize_observation(
        upper_obs_space_structured_full) == upper_100_structured_norm).all()
    assert (os_structured_full.normalize_observation(
        lower_obs_space_structured_full) == lower_100_structured_norm).all()

    assert (os_custom_keys_norm.normalize_observation(upper_obs_space_custom) ==
            1).all()
    assert (os_custom_keys_norm.normalize_observation(lower_obs_space_custom) ==
            -1).all()


def test_denormalize_observation(os_structured_full, os_custom_keys_norm):
    assert os_structured_full.denormalize_observation(np.array(
        [1.0] * 27)) == pytest.approx(upper_obs_space_structured_full)
    assert os_structured_full.denormalize_observation(np.array(
        [-1.0] * 27)) == pytest.approx(lower_obs_space_structured_full)

    assert os_custom_keys_norm.denormalize_observation(np.array(
        [1.0] * 18)) == pytest.approx(upper_obs_space_custom)
    assert os_custom_keys_norm.denormalize_observation(np.array(
        [-1.0] * 18)) == pytest.approx(lower_obs_space_custom)


def test_satisfy_constraints(os_default, os_structured_full):
    assert os_default.satisfy_constraints(upper_99_structured_norm)
    assert not os_default.satisfy_constraints(upper_100_structured_norm)
    assert not os_default.satisfy_constraints(upper_101_structured_norm)

    assert os_structured_full.satisfy_constraints(
        lower_obs_space_structured_full * 0.5)
    assert not os_structured_full.satisfy_constraints(
        lower_obs_space_structured_full * 1.01)

    assert os_default.satisfy_constraints(lower_99_structured_norm)
    assert not os_default.satisfy_constraints(lower_100_structured_norm)
    assert not os_default.satisfy_constraints(lower_101_structured_norm)


def test_clip_observation(os_default, os_structured_full):
    assert (os_default.clip_observation(upper_99_structured_norm) ==
            upper_99_structured_norm).all()
    assert (os_default.clip_observation(lower_99_structured_norm) ==
            lower_99_structured_norm).all()
    assert (os_default.clip_observation(upper_100_structured_norm) ==
            upper_100_structured_norm).all()
    assert (os_default.clip_observation(lower_100_structured_norm) ==
            lower_100_structured_norm).all()
    assert (os_default.clip_observation(upper_101_structured_norm) ==
            upper_100_structured_norm).all()
    assert (os_default.clip_observation(lower_101_structured_norm) ==
            lower_100_structured_norm).all()

    assert os_structured_full.clip_observation(
        os_structured_full.denormalize_observation(
            normalized_struct_obs_to_be_clipped)) == pytest.approx(
                os_structured_full.denormalize_observation(
                    normalized_struct_obs_clipped))


def test_add_and_remove_observation(os_custom_keys_norm):
    os_custom_keys_norm.add_observation("joint_torques")
    assert os_custom_keys_norm._observations_keys == [
        "end_effector_positions", "action_joint_positions", "joint_torques"
    ]
    assert len(os_custom_keys_norm.get_observation_spaces().low) == 27
    assert len(os_custom_keys_norm.get_observation_spaces().high) == 27
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    os_custom_keys_norm.remove_observations(["joint_torques"])
    assert os_custom_keys_norm._observations_keys == [
        "end_effector_positions", "action_joint_positions"
    ]
    assert len(os_custom_keys_norm.get_observation_spaces().low) == 18
    assert len(os_custom_keys_norm.get_observation_spaces().high) == 18
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()

    def dummy_fn(robot):
        return [1, 1, 1]

    os_custom_keys_norm.add_observation("dummy_key",
                                        lower_bound=[-4, -4, -4],
                                        upper_bound=[2, 2, 2],
                                        observation_fn=dummy_fn)
    assert os_custom_keys_norm._observations_keys == [
        "end_effector_positions", "action_joint_positions", "dummy_key"
    ]
    assert len(os_custom_keys_norm.get_observation_spaces().low) == 21
    assert len(os_custom_keys_norm.get_observation_spaces().high) == 21
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    os_custom_keys_norm.remove_observations(["dummy_key"])
    assert os_custom_keys_norm._observations_keys == [
        "end_effector_positions", "action_joint_positions"
    ]
    assert len(os_custom_keys_norm.get_observation_spaces().low) == 18
    assert len(os_custom_keys_norm.get_observation_spaces().high) == 18
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
    assert (os_custom_keys_norm.get_observation_spaces().high == 1.).all()
