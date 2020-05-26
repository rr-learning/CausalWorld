from stable_baselines import PPO2
from causal_rl_bench.agents.policy_base import PolicyBase
import os


class DummpyPolicy(PolicyBase):
    def __init__(self):
        super(DummpyPolicy, self).__init__()
        self.action = None
        return

    def act(self, obs):
        return self.action

    def add_action(self, action):
        self.action = action
        return
