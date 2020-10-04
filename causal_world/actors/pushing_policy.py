from causal_world.actors.base_policy import BaseActorPolicy
import os
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2
except ImportError:
    pass


class PushingActorPolicy(BaseActorPolicy):

    def __init__(self):
        """
        This policy is expected to run @83.3 Hz.
        The policy expects normalized observations and it outputs
        desired joint positions.

        - This policy is trained with several goals.
        """
        #TODO: replace with find catkin
        super(PushingActorPolicy, self).__init__('pushing_policy')
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "../assets/baseline_actors"
                            "/pushing_ppo_curr1.zip")
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
