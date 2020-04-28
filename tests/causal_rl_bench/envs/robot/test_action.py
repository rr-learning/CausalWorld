from causal_rl_bench.envs.robot.action import TriFingerAction

import numpy as np
import pytest


@pytest.fixture(scope='module')
def as_jp_norm():
    return TriFingerAction(action_mode="joint_positions", normalize_actions=True)


@pytest.fixture(scope='module')
def as_jp_full():
    return TriFingerAction(action_mode="joint_positions", normalize_actions=False)


@pytest.fixture(scope='module')
def as_jt_norm():
    return TriFingerAction(action_mode="joint_torques", normalize_actions=True)


@pytest.fixture(scope='module')
def as_jt_full():
    return TriFingerAction(action_mode="joint_torques", normalize_actions=False)


@pytest.fixture(scope='module')
def as_default():
    return TriFingerAction()


upper_99_normalized_action = np.array([0.99] * 9)
upper_100_normalized_action = np.array([1.0] * 9)
upper_101_normalized_action = np.array([1.01] * 9)

lower_99_normalized_action = np.array([-0.99] * 9)
lower_100_normalized_action = np.array([-1.0] * 9)
lower_101_normalized_action = np.array([-1.01] * 9)


@pytest.mark.skip
def test_set_action_space():
    assert False


@pytest.mark.skip
def test_get_action_space(as_default, as_jt_full, as_jt_norm, as_jp_full, as_jp_norm):
    assert as_default.get_action_space().low == -1


def test_is_normalized(as_default, as_jt_full, as_jt_norm, as_jp_full, as_jp_norm):
    assert as_default.is_normalized()
    assert as_jt_norm.is_normalized()
    assert not as_jt_full.is_normalized()
    assert as_jt_norm.is_normalized()
    assert not as_jp_full.is_normalized()


def test_satisfy_constraints(as_default, as_jt_full, as_jt_norm, as_jp_full, as_jp_norm):
    assert as_default.satisfy_constraints(upper_99_normalized_action)
    assert not as_default.satisfy_constraints(upper_100_normalized_action)
    assert not as_default.satisfy_constraints(upper_101_normalized_action)

    assert as_jt_norm.satisfy_constraints(upper_99_normalized_action)
    assert not as_jt_norm.satisfy_constraints(upper_100_normalized_action)
    assert not as_jt_norm.satisfy_constraints(upper_101_normalized_action)

    assert as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(upper_99_normalized_action))
    assert not as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(upper_100_normalized_action))
    assert not as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(upper_101_normalized_action))

    assert as_jp_norm.satisfy_constraints(upper_99_normalized_action)
    assert not as_jp_norm.satisfy_constraints(upper_100_normalized_action)
    assert not as_jp_norm.satisfy_constraints(upper_101_normalized_action)

    assert as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(upper_99_normalized_action))
    assert not as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(upper_100_normalized_action))
    assert not as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(upper_101_normalized_action))

    assert as_default.satisfy_constraints(lower_99_normalized_action)
    assert not as_default.satisfy_constraints(lower_100_normalized_action)
    assert not as_default.satisfy_constraints(lower_101_normalized_action)

    assert as_jt_norm.satisfy_constraints(lower_99_normalized_action)
    assert not as_jt_norm.satisfy_constraints(lower_100_normalized_action)
    assert not as_jt_norm.satisfy_constraints(lower_101_normalized_action)

    assert as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(lower_99_normalized_action))
    assert not as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(lower_100_normalized_action))
    assert not as_jt_full.satisfy_constraints(as_jt_full.denormalize_action(lower_101_normalized_action))

    assert as_jp_norm.satisfy_constraints(lower_99_normalized_action)
    assert not as_jp_norm.satisfy_constraints(lower_100_normalized_action)
    assert not as_jp_norm.satisfy_constraints(lower_101_normalized_action)

    assert as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(lower_99_normalized_action))
    assert not as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(lower_100_normalized_action))
    assert not as_jp_full.satisfy_constraints(as_jp_full.denormalize_action(lower_101_normalized_action))


def test_clip_action(as_default, as_jt_full, as_jt_norm, as_jp_full, as_jp_norm):
    assert (as_default.clip_action(upper_99_normalized_action) == upper_99_normalized_action).all()
    assert (as_default.clip_action(lower_99_normalized_action) == lower_99_normalized_action).all()
    assert (as_default.clip_action(upper_100_normalized_action) == upper_100_normalized_action).all()
    assert (as_default.clip_action(lower_100_normalized_action) == lower_100_normalized_action).all()
    assert (as_default.clip_action(upper_101_normalized_action) == upper_100_normalized_action).all()
    assert (as_default.clip_action(lower_101_normalized_action) == lower_100_normalized_action).all()

    normalized_action_to_be_clipped = np.array([-1.01, 2.0, 0.5, 0.4, 1.3, -6.0, 0.0, 0.3, 1.1])
    normalized_action_clipped = np.array([-1.0, 1.0, 0.5, 0.4, 1.0, -1.0, 0.0, 0.3, 1.0])

    assert (as_default.clip_action(normalized_action_to_be_clipped) == normalized_action_clipped).all()
    assert (as_jt_norm.clip_action(normalized_action_to_be_clipped) == normalized_action_clipped).all()
    assert (as_jp_norm.clip_action(normalized_action_to_be_clipped) == normalized_action_clipped).all()
    assert as_jp_full.clip_action(as_jp_full.denormalize_action(normalized_action_to_be_clipped)) == pytest.approx(as_jp_full.denormalize_action(normalized_action_clipped))


@pytest.mark.skip
def test_normalize_action():
    assert False


@pytest.mark.skip
def test_denormalize_action():
    assert False
