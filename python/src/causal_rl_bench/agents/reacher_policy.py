from stable_baselines import PPO2
import os


class ReacherPolicy(object):
    def __init__(self):
        #TODO: replace with find catkin
        self.model = PPO2.load(os.path.join(
            os.path.dirname(os.path.
                            realpath(__file__)),
            "../../../assets/reacher_model.zip"))
        return

    def act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]
