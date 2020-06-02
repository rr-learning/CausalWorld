from stable_baselines import PPO2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from causal_rl_bench.agents.base_policy import BaseActorPolicy
import os


class ReacherActorPolicy(BaseActorPolicy):
    def __init__(self):
        """
        This policy is expected to run @250 Hz the inputs order to the policy
        are as follows:
       =["joint_positions", "joint_velocities", "end_effector_positions",
         "action_joint_positions", "end_effector_positions_goal"]
         -> desired absolute joint actions
        """
        #TODO: replace with find catkin
        super(ReacherActorPolicy, self).__init__()
        self.model = PPO2.load(os.path.join(
            os.path.dirname(os.path.
                            realpath(__file__)),
            "../../../assets/reacher_model.zip"))
        return

    def act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]
