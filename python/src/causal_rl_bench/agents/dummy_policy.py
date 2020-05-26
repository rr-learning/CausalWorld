from causal_rl_bench.agents.policy_base import PolicyBase


class DummyPolicy(PolicyBase):
    def __init__(self):
        super(DummyPolicy, self).__init__()
        self.action = None
        return

    def act(self, obs):
        return self.action

    def add_action(self, action):
        self.action = action
        return
