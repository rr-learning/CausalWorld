class BaseActorPolicy(object):
    """
    This is a policy wrapper for an actor, its functions need to be filled
    to load the policy such that it can be used by the robot to act in the
    environment.
    """

    def __init__(self, identifier=None):
        """

        :param identifier: (str) defines the name of the actor policy
        """
        self.identifier = identifier
        return

    def get_identifier(self):
        """
        :return: (str) defines the name of the actor policy
        """
        return self.identifier

    def act(self, obs):
        """
        The function is called for the agent to act in the world.

        :param obs: (nd.array) defines the observations received by the agent
                               at time step t

        :return: (nd.array) defines the action to be executed at time step t
        """
        raise NotImplementedError()

    def reset(self):
        """
        The function is called for the controller to be cleared.

        :return:
        """
        return
