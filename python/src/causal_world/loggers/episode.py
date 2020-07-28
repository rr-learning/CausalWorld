class Episode:
    def __init__(self,
                 task_name,
                 initial_full_state,
                 task_params=None,
                 world_params=None):
        """
        The structure in which the data from each episode
        will be logged.

        :param task_name:
        :param initial_full_state:
        :param task_params:
        :param world_params:
        """
        self.task_name = task_name
        self.task_params = task_params
        self.world_params = world_params
        self.initial_full_state = initial_full_state
        self.robot_actions = []
        self.observations = []
        self.rewards = []
        self.infos = []
        self.dones = []
        self.timestamps = []

    def append(self, robot_action, observation, reward, info, done, timestamp):
        """

        :param robot_action:
        :param observation:
        :param reward:
        :param info:
        :param done:
        :param timestamp:

        :return:
        """
        self.robot_actions.append(robot_action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.infos.append(info)
        self.dones.append(done)
