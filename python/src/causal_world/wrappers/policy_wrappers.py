from causal_world.actors.base_policy import BaseActorPolicy


class MovingAverageActionWrapperActorPolicy(BaseActorPolicy):

    def __init__(self, policy, widow_size=8, initial_value=0):
        """

        :param policy:
        :param widow_size:
        :param initial_value:
        """
        super(BaseActorPolicy, self).__init__()
        self.__widow_size = widow_size
        self.__buffer = [initial_value / widow_size] * widow_size
        self.__avg = initial_value
        self.__p = 0
        self.__start_smoothing = False
        self.__initial_counter = 0
        self.__policy = policy

    @property
    def avg(self):
        """

        :return: 
        """
        return self.__avg

    @property
    def policy(self):
        """

        :return:
        """
        return self.__policy

    def act(self, observation):
        """

        :param observation:
        :return:
        """
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
