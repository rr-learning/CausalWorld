from causal_world.actors.base_policy import BaseActorPolicy


class DummyActorPolicy(BaseActorPolicy):
    """
    This is a policy wrapper for a dummy actor, which uses the interface of
    the actor policy but is basically fed the actions externally, (i.e just
    using the interface of the actor policy but actions are calculated
    externally)
    """
    def __init__(self):
        super(DummyActorPolicy, self).__init__(identifier="dummy_policy")
        self.action = None
        return

    def act(self, obs):
        """
        The function is called for the agent to act in the world.

        :param obs: (nd.array) defines the observations received by the agent
                               at time step t

        :return: (nd.array) defines the action to be executed at time step t
        """
        return self.action

    def add_action(self, action):
        """
        The function used to add actions which would be returned further when
        the act function is called. Can be used if the actions are calculated
        externally.

        :param action: (nd.array) defines the action to be executed at time
                                  step t

        :return:
        """
        self.action = action
        return

