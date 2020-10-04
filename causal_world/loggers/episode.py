import copy

class Episode:
    def __init__(self,
                 task_name,
                 initial_full_state,
                 task_params=None,
                 world_params=None):
        """
        The structure in which the data from each episode
        will be logged.

        :param task_name: (str) task generator name
        :param initial_full_state: (dict) dict specifying the full state
                                          variables of the environment.
        :param task_params: (dict) task generator parameters.
        :param world_params: (dict) causal world parameters.
        """
        self.task_name = task_name
        self.task_params = copy.deepcopy(task_params)
        if 'task_name' in self.task_params:
            del self.task_params['task_name']
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

        :param robot_action: (nd.array) action passed to step function.
        :param observation: (nd.array) observations returned after stepping
                                       through the environment.
        :param reward: (float) reward received from the environment.
        :param info: (dict) dictionary specifying all the extra information
                            after stepping through the environment.
        :param done: (bool) true if the environment returns done.
        :param timestamp: (float) time stamp with respect to the beginning of
                                  the episode.

        :return:
        """
        self.robot_actions.append(robot_action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.infos.append(info)
        self.dones.append(done)
        return
