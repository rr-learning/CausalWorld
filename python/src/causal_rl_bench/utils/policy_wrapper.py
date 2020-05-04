class PolicyWrapper:
    """
    This is a policy wrapper template class whose methods needs to
    be implemented to record and view a trained policy using the
    TaskViewer class
    """
    def __init__(self):
        pass

    def get_identifier(self):
        raise NotImplementedError()

    def get_action_for_observation(self, observation):
        raise NotImplementedError()
