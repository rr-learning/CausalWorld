from causal_world.envs.robot.action import TriFingerAction

import math
import numpy as np
import pytest


@pytest.fixture(scope='module')
def as_jp_norm():
    return TriFingerAction(action_mode="joint_positions",
                           normalize_actions=True)


@pytest.fixture(scope='module')
def as_jp_full():
    return TriFingerAction(action_mode="joint_positions",
                           normalize_actions=False)


@pytest.fixture(scope='module')
def as_jt_norm():
    return TriFingerAction(action_mode="joint_torques", normalize_actions=True)


@pytest.fixture(scope='module')
def as_jt_full():
    return TriFingerAction(action_mode="joint_torques", normalize_actions=False)


@pytest.fixture(scope='module')
def as_default():
    return TriFingerAction()


@pytest.fixture(scope='module')
def as_custom():
    return TriFingerAction(normalize_actions=False)


upper_99_normalized_action = np.array([0.99] * 9)
upper_100_normalized_action = np.array([1.0] * 9)
upper_101_normalized_action = np.array([1.01] * 9)

lower_99_normalized_action = np.array([-0.99] * 9)
lower_100_normalized_action = np.array([-1.0] * 9)
lower_101_normalized_action = np.array([-1.01] * 9)

upper_100_denormalized_jt_action = np.array([0.36] * 9)
lower_100_denormalized_jt_action = np.array([-0.36] * 9)

lower_100_denormalized_jp_action = np.array(
    [-1.57, -1.2, -3.0] * 3)
upper_100_denormalized_jp_action = np.array(
    [1.0, 1.57, 3.0] * 3)

normalized_action_to_be_clipped = np.array(
    [-1.01, 2.0, 0.5, 0.4, 1.3, -6.0, 0.0, 0.3, 1.1])
normalized_action_clipped = np.array(
    [-1.0, 1.0, 0.5, 0.4, 1.0, -1.0, 0.0, 0.3, 1.0])

custom_action_upper_bound = np.array([1, 2, 3, 4])
custom_action_lower_bound = np.array([0, 1, -3, -10])

custom_action_norm_upper_bound = np.array([1, 1, 1, 1])
custom_action_norm_lower_bound = np.array([-1, -1, -1, -1])


def test_set_action_space(as_custom):
    as_custom.set_action_space(custom_action_lower_bound,
                               custom_action_upper_bound)
    assert (as_custom.get_action_space().low == custom_action_lower_bound).all()
    assert (
        as_custom.get_action_space().high == custom_action_upper_bound).all()
    assert (as_custom.normalize_action(custom_action_upper_bound) ==
            custom_action_norm_upper_bound).all()
    assert (as_custom.normalize_action(custom_action_lower_bound) ==
            custom_action_norm_lower_bound).all


def test_get_action_space(as_default, as_jt_full, as_jt_norm, as_jp_full,
                          as_jp_norm):
    assert (as_default.get_action_space().low == -1.).all()
    assert (as_jp_norm.get_action_space().low == -1.).all()
    assert (as_jt_norm.get_action_space().low == -1.).all()
    assert (as_default.get_action_space().high == 1.).all()
    assert (as_jp_norm.get_action_space().high == 1.).all()
    assert (as_jt_norm.get_action_space().high == 1.).all()

    assert (as_jp_full.get_action_space().low ==
            lower_100_denormalized_jp_action).all()
    assert (as_jt_full.get_action_space().low ==
            lower_100_denormalized_jt_action).all()
    assert (as_jp_full.get_action_space().high ==
            upper_100_denormalized_jp_action).all()
    assert (as_jt_full.get_action_space().high ==
            upper_100_denormalized_jt_action).all()


def test_is_normalized(as_default, as_jt_full, as_jt_norm, as_jp_full,
                       as_jp_norm):
    assert as_default.is_normalized()
    assert as_jt_norm.is_normalized()
    assert not as_jt_full.is_normalized()
    assert as_jt_norm.is_normalized()
    assert not as_jp_full.is_normalized()


def test_satisfy_constraints(as_default, as_jt_full, as_jt_norm, as_jp_full,
                             as_jp_norm):
    assert as_default.satisfy_constraints(upper_99_normalized_action)
    assert not as_default.satisfy_constraints(upper_100_normalized_action)
    assert not as_default.satisfy_constraints(upper_101_normalized_action)

    assert as_jt_norm.satisfy_constraints(upper_99_normalized_action)
    assert not as_jt_norm.satisfy_constraints(upper_100_normalized_action)
    assert not as_jt_norm.satisfy_constraints(upper_101_normalized_action)

    assert as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(upper_99_normalized_action))
    assert not as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(upper_100_normalized_action))
    assert not as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(upper_101_normalized_action))

    assert as_jp_norm.satisfy_constraints(upper_99_normalized_action)
    assert not as_jp_norm.satisfy_constraints(upper_100_normalized_action)
    assert not as_jp_norm.satisfy_constraints(upper_101_normalized_action)

    assert as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(upper_99_normalized_action))
    assert not as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(upper_100_normalized_action))
    assert not as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(upper_101_normalized_action))

    assert as_default.satisfy_constraints(lower_99_normalized_action)
    assert not as_default.satisfy_constraints(lower_100_normalized_action)
    assert not as_default.satisfy_constraints(lower_101_normalized_action)

    assert as_jt_norm.satisfy_constraints(lower_99_normalized_action)
    assert not as_jt_norm.satisfy_constraints(lower_100_normalized_action)
    assert not as_jt_norm.satisfy_constraints(lower_101_normalized_action)

    assert as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(lower_99_normalized_action))
    assert not as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(lower_100_normalized_action))
    assert not as_jt_full.satisfy_constraints(
        as_jt_full.denormalize_action(lower_101_normalized_action))

    assert as_jp_norm.satisfy_constraints(lower_99_normalized_action)
    assert not as_jp_norm.satisfy_constraints(lower_100_normalized_action)
    assert not as_jp_norm.satisfy_constraints(lower_101_normalized_action)

    assert as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(lower_99_normalized_action))
    assert not as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(lower_100_normalized_action))
    assert not as_jp_full.satisfy_constraints(
        as_jp_full.denormalize_action(lower_101_normalized_action))


def test_clip_action(as_default, as_jt_full, as_jt_norm, as_jp_full,
                     as_jp_norm):
    assert (as_default.clip_action(upper_99_normalized_action) ==
            upper_99_normalized_action).all()
    assert (as_default.clip_action(lower_99_normalized_action) ==
            lower_99_normalized_action).all()
    assert (as_default.clip_action(upper_100_normalized_action) ==
            upper_100_normalized_action).all()
    assert (as_default.clip_action(lower_100_normalized_action) ==
            lower_100_normalized_action).all()
    assert (as_default.clip_action(upper_101_normalized_action) ==
            upper_100_normalized_action).all()
    assert (as_default.clip_action(lower_101_normalized_action) ==
            lower_100_normalized_action).all()

    assert (as_default.clip_action(normalized_action_to_be_clipped) ==
            normalized_action_clipped).all()
    assert (as_jt_norm.clip_action(normalized_action_to_be_clipped) ==
            normalized_action_clipped).all()
    assert (as_jp_norm.clip_action(normalized_action_to_be_clipped) ==
            normalized_action_clipped).all()
    assert as_jp_full.clip_action(
        as_jp_full.denormalize_action(
            normalized_action_to_be_clipped)) == pytest.approx(
                as_jp_full.denormalize_action(normalized_action_clipped))


def test_normalize_action(as_default, as_jt_full, as_jt_norm, as_jp_full,
                          as_jp_norm):
    assert (as_default.normalize_action(upper_100_denormalized_jp_action) ==
            upper_100_normalized_action).all()
    assert (as_jp_full.normalize_action(upper_100_denormalized_jp_action) ==
            upper_100_normalized_action).all()
    assert (as_jp_norm.normalize_action(upper_100_denormalized_jp_action) ==
            upper_100_normalized_action).all()
    assert (as_jt_full.normalize_action(upper_100_denormalized_jt_action) ==
            upper_100_normalized_action).all()
    assert (as_jt_norm.normalize_action(upper_100_denormalized_jt_action) ==
            upper_100_normalized_action).all()
    assert (as_jt_full.normalize_action(upper_100_denormalized_jp_action) !=
            upper_100_normalized_action).all()
    assert (as_jt_norm.normalize_action(upper_100_denormalized_jp_action) !=
            upper_100_normalized_action).all()
    assert (as_jp_full.normalize_action(upper_100_denormalized_jt_action) !=
            upper_100_normalized_action).all()
    assert (as_jp_norm.normalize_action(upper_100_denormalized_jt_action) !=
            upper_100_normalized_action).all()

    assert (as_default.normalize_action(lower_100_denormalized_jp_action) ==
            lower_100_normalized_action).all()
    assert (as_jp_full.normalize_action(lower_100_denormalized_jp_action) ==
            lower_100_normalized_action).all()
    assert (as_jp_norm.normalize_action(lower_100_denormalized_jp_action) ==
            lower_100_normalized_action).all()
    assert (as_jt_full.normalize_action(lower_100_denormalized_jt_action) ==
            lower_100_normalized_action).all()
    assert (as_jt_norm.normalize_action(lower_100_denormalized_jt_action) ==
            lower_100_normalized_action).all()
    assert (as_jt_full.normalize_action(lower_100_denormalized_jp_action) !=
            lower_100_normalized_action).all()
    assert (as_jt_norm.normalize_action(lower_100_denormalized_jp_action) !=
            lower_100_normalized_action).all()
    assert (as_jp_full.normalize_action(lower_100_denormalized_jt_action) !=
            lower_100_normalized_action).all()
    assert (as_jp_norm.normalize_action(lower_100_denormalized_jt_action) !=
            lower_100_normalized_action).all()

    # convert back and forth
    assert as_jp_norm.denormalize_action(
        as_jp_norm.normalize_action(upper_100_denormalized_jp_action)
    ) == pytest.approx(upper_100_denormalized_jp_action)
    assert as_jt_norm.denormalize_action(
        as_jt_norm.normalize_action(upper_100_denormalized_jt_action)
    ) == pytest.approx(upper_100_denormalized_jt_action)
    assert as_jp_norm.denormalize_action(
        as_jp_norm.normalize_action(lower_100_denormalized_jp_action)
    ) == pytest.approx(lower_100_denormalized_jp_action)
    assert as_jt_norm.denormalize_action(
        as_jt_norm.normalize_action(lower_100_denormalized_jt_action)
    ) == pytest.approx(lower_100_denormalized_jt_action)


def test_denormalize_action(as_default, as_jt_full, as_jt_norm, as_jp_full,
                            as_jp_norm):
    assert as_default.denormalize_action(
        upper_100_normalized_action) == pytest.approx(
            upper_100_denormalized_jp_action)
    assert as_jp_full.denormalize_action(
        upper_100_normalized_action) == pytest.approx(
            upper_100_denormalized_jp_action)
    assert as_jp_norm.denormalize_action(
        upper_100_normalized_action) == pytest.approx(
            upper_100_denormalized_jp_action)
    assert as_jt_norm.denormalize_action(
        upper_100_normalized_action) == pytest.approx(
            upper_100_denormalized_jt_action)
    assert as_jt_full.denormalize_action(
        upper_100_normalized_action) == pytest.approx(
            upper_100_denormalized_jt_action)
    assert (as_jt_norm.denormalize_action(upper_100_normalized_action) !=
            upper_100_denormalized_jp_action).all()
    assert (as_jt_full.denormalize_action(upper_100_normalized_action) !=
            upper_100_denormalized_jp_action).all()
    assert (as_jp_full.denormalize_action(upper_100_normalized_action) !=
            upper_100_denormalized_jt_action).all()
    assert (as_jp_norm.denormalize_action(upper_100_normalized_action) !=
            upper_100_denormalized_jt_action).all()

    assert as_default.denormalize_action(
        lower_100_normalized_action) == pytest.approx(
            lower_100_denormalized_jp_action)
    assert as_jp_full.denormalize_action(
        lower_100_normalized_action) == pytest.approx(
            lower_100_denormalized_jp_action)
    assert as_jp_norm.denormalize_action(
        lower_100_normalized_action) == pytest.approx(
            lower_100_denormalized_jp_action)
    assert as_jt_full.denormalize_action(
        lower_100_normalized_action) == pytest.approx(
            lower_100_denormalized_jt_action)
    assert as_jt_norm.denormalize_action(
        lower_100_normalized_action) == pytest.approx(
            lower_100_denormalized_jt_action)
    assert (as_jt_full.denormalize_action(lower_100_normalized_action) !=
            lower_100_denormalized_jp_action).all()
    assert (as_jt_norm.denormalize_action(lower_100_normalized_action) !=
            lower_100_denormalized_jp_action).all()
    assert (as_jp_full.denormalize_action(lower_100_normalized_action) !=
            lower_100_denormalized_jt_action).all()
    assert (as_jp_norm.denormalize_action(lower_100_normalized_action) !=
            lower_100_denormalized_jt_action).all()

    # convert back and forth
    assert as_jp_norm.denormalize_action(
        as_jp_norm.normalize_action(upper_100_denormalized_jp_action)
    ) == pytest.approx(upper_100_denormalized_jp_action)
    assert as_jt_norm.denormalize_action(
        as_jt_norm.normalize_action(upper_100_denormalized_jt_action)
    ) == pytest.approx(upper_100_denormalized_jt_action)
    assert as_jp_norm.denormalize_action(
        as_jp_norm.normalize_action(lower_100_denormalized_jp_action)
    ) == pytest.approx(lower_100_denormalized_jp_action)
    assert as_jt_norm.denormalize_action(
        as_jt_norm.normalize_action(lower_100_denormalized_jt_action)
    ) == pytest.approx(lower_100_denormalized_jt_action)
