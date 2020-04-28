from causal_rl_bench.envs.robot.action import TriFingerAction

import numpy as np


class TestTriFingerAction:
    def test_default_action_space(self):
        default_action_space = TriFingerAction()
        upper_normalized_action = np.array([0.99] * 9)
        # upper_denormalized_action = default_action_space.denormalize_action(upper_normalized_action)
        assert default_action_space.is_normalized(), "Default action space not normalized"
        assert default_action_space.satisfy_constraints(upper_normalized_action), "Constraints satisfied"
        # assert not default_action_space.satisfy_constraints(upper_denormalized_action), "Constraints satisfied"

    def test_set_action_space(self):
        assert False
