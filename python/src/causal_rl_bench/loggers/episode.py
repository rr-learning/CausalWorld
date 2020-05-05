class Episode:
    """
        The structure in which the data from each episode
        will be logged.
        """

    def __init__(self, task_params, initial_full_state):
        self.task_params = task_params
        self.initial_full_state = initial_full_state
        self.robot_actions = []
        self.observations = []
        self.rewards = []
        self.timestamps = []

    def append(self, robot_action, observation, reward, timestamp):
        self.robot_actions.append(robot_action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
