from causal_rl_bench.agents.base_policy import BaseActorPolicy


class DummyActorPolicy(BaseActorPolicy):
    def __init__(self):
        super(DummyActorPolicy, self).__init__()
        self.action = None
        return

    def act(self, obs):
        return self.action

    def add_action(self, action):
        self.action = action
        return
