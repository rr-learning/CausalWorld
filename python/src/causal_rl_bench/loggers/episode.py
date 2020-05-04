class Episode:
    """
        The structure in which the data from each episode
        will be logged.
        """

    def __init__(self, task_params):
        self.task_params = task_params
        self.robot_actions = []
        self.world_states = []
        self.rewards = []
        self.timestamps = []

    def append(self, robot_action, world_state, reward, timestamp):
        self.robot_actions.append(robot_action)
        self.world_states.append(world_state)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
