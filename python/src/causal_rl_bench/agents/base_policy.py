class BaseActorPolicy(object):
    """
    This is a policy wrapper template class whose methods needs to
    be implemented to record and view a trained policy using the
    TaskViewer class
    """
    def __init__(self, identifier=None):
        self.identifier = identifier
        return

    def get_identifier(self):
        return self.identifier

    def act(self, obs):
        raise NotImplementedError()
