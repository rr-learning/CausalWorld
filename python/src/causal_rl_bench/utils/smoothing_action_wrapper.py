from causal_rl_bench.agents.policy_base import PolicyBase
from causal_rl_bench.agents.dummy_policy import DummpyPolicy
import gym


class MovingAverageActionPolicyWrapper(PolicyBase):
    def __init__(self, policy, widow_size=8, initial_value=0):
        super(PolicyBase, self).__init__()
        self.__widow_size = widow_size
        self.__buffer = [initial_value / widow_size] * widow_size
        self.__avg = initial_value
        self.__p = 0
        self.__start_smoothing = False
        self.__initial_counter = 0
        self.__policy = policy

    @property
    def avg(self):
        """Returns current moving average value"""
        return self.__avg

    @property
    def policy(self):
        """Returns current moving average value"""
        return self.__policy

    def act(self, observation):
        unsmoothed_action = self.__policy.act(observation)
        self.__avg -= self.__buffer[self.__p]
        self.__buffer[self.__p] = unsmoothed_action / self.__widow_size
        self.__avg += self.__buffer[self.__p]
        self.__p = (self.__p + 1) % self.__widow_size
        if not self.__start_smoothing:
            self.__initial_counter += 1
            if self.__initial_counter >= self.__widow_size:
                self.__start_smoothing = True
        if self.__start_smoothing:
            return self.__avg
        else:
            return unsmoothed_action


class MovingAverageActionEnvWrapper(gym.ActionWrapper):
    def __init__(self, env, widow_size=8, initial_value=0):
        super(MovingAverageActionEnvWrapper, self).__init__(env)
        self.__policy = DummpyPolicy()
        self.__policy = MovingAverageActionPolicyWrapper(self.__policy,
                                                         widow_size=widow_size,
                                                         initial_value=initial_value)
        return

    def action(self, action):
        self.__policy.policy.add_action(action) #hack now
        return self.__policy.act(observation=None)

    def reverse_action(self, action):
        raise Exception("not implemented yet")
