from causal_world.actors.base_policy import BaseActorPolicy
import os
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2
except ImportError:
    pass


class PickAndPlaceActorPolicy(BaseActorPolicy):

    def __init__(self):
        """
        This policy is expected to run @83.3 Hz.
        The policy expects normalized observations and it outputs
        desired joint positions.

        - This policy is trained with one goal positions only.
        """
        #TODO: replace with find catkin
        super(PickAndPlaceActorPolicy, self).__init__('pick_and_place_policy')
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "../assets/baseline_actors"
                            "/pick_and_place_ppo_curr0.zip")
        self._policy = PPO2.load(file)
        return

    def act(self, obs):
        """
        The function is called for the agent to act in the world.

        :param obs: (nd.array) defines the observations received by the agent
                               at time step t

        :return: (nd.array) defines the action to be executed at time step t
        """
        return self._policy.predict(obs, deterministic=True)[0]
