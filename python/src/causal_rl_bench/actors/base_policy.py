class BaseActorPolicy(object):
    """
    This is a policy wrapper template class whose methods needs to
    be implemented to record and view a trained policy using the
    TaskViewer class
    """

    def __init__(self, identifier=None):
        """

        :param identifier:
        """
        self.identifier = identifier
        return

    def get_identifier(self):
        """

        :return:
        """
        return self.identifier

    def act(self, obs):
        """

        :param obs:
        :return:
        """
        raise NotImplementedError()
