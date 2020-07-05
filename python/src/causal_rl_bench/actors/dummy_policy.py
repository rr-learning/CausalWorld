from causal_rl_bench.actors.base_policy import BaseActorPolicy


class DummyActorPolicy(BaseActorPolicy):
    def __init__(self):
        super(DummyActorPolicy, self).__init__()
        self.action = None
        return

    def act(self, obs):
        """

        :param obs:
        :return:
        """
        return self.action

    def add_action(self, action):
        """

        :param action:
        :return:
        """
        self.action = action
        return
