from causal_world.actors.base_policy import BaseActorPolicy
import numpy as np


class RandomActorPolicy(BaseActorPolicy):
    """
    This is a policy wrapper for a random actor.
    """
    def __init__(self, low_bound, upper_bound):
        super(RandomActorPolicy, self).__init__(identifier="random_policy")
        self._low_bound = low_bound
        self._upper_bound = upper_bound
        return

    def act(self, obs):
        """
        The function is called for the agent to act in the world.

        :param obs: (nd.array) defines the observations received by the agent
                               at time step t

        :return: (nd.array) defines the action to be executed at time step t
        """
        return np.random.uniform(self._low_bound, self._upper_bound)

