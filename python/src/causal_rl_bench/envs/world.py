import numpy as np
import gym
from causal_rl_bench.envs.robot.trifinger import TriFingerRobot
from causal_rl_bench.loggers.world_logger import WorldLogger
from causal_rl_bench.envs.scene.stage import Stage
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.envs.env_utils import combine_spaces


class World(gym.Env):
    def __init__(self, task=None, task_id="picking", skip_frame=0.02,
                 enable_visualization=True, seed=0,
                 action_mode="joint_positions", observation_mode="structured",
                 camera_skip_frame=0.3, normalize_actions=True,
                 normalize_observations=True, max_episode_length=None,
                 logging=False, **kwargs):
        """
        Constructor sets up the physical world parameters,
        and resets to begin training.

        Args:
            skip_frame (float): the time step at which the env step runs
            finger-type (str "single"/"tri"): to train on the "single"
                or the "tri" finger
            enable_visualization (bool): if the simulation env is to be
                visualized
        """
        self.robot = TriFingerRobot(action_mode=action_mode,
                                    observation_mode=observation_mode,
                                    enable_visualization=enable_visualization,
                                    skip_frame=skip_frame,
                                    camera_skip_frame=camera_skip_frame,
                                    normalize_actions=normalize_actions,
                                    normalize_observations=normalize_observations
                                    )
        self.seed(seed)
        self.stage = Stage(observation_mode=observation_mode,
                           normalize_observations=normalize_observations)
        if max_episode_length is None:
            self.enforce_episode_length = False
        else:
            self.enforce_episode_length = True
        self.max_episode_length = max_episode_length
        self.episode_length = 0

        self.metadata = {"render.modes": ["human"]}
        gym.Env.__init__(self)
        if task is None:
            if task_id == "picking":
                self.task = PickingTask()
            elif task_id == "pushing":
                self.task = PushingTask()
        else:
            self.task = task
        self.task.init_task(self.robot, self.stage)
        selected_observations = self.task.observation_keys
        self.robot.select_observations(selected_observations)
        self.stage.select_observations(selected_observations)
        self.observation_space = \
            combine_spaces(self.robot.get_observation_spaces(),
                           self.stage.get_observation_spaces())
        self.action_space = self.robot.get_action_spaces()
        self.logging = logging
        self.logger = WorldLogger()
        self.skip_frame = skip_frame
        #TODO: verify spaces here
        self.max_time_steps = 5000
        return

    def step(self, action):
        self.episode_length += 1

        robot_observations_dict = self.robot.apply_action(action)
        stage_observations_dict = self.stage.get_current_observations()
        task_observations = self.task.filter_observations(robot_observations_dict,
                                                          stage_observations_dict)
        reward = self.task.get_reward()
        done = self._is_done()
        info = {}

        if self.logging:
            # TODO: pass the full state to the logger in the future
            self.logger.append(robot_action=action,
                               world_state=stage_observations_dict,
                               reward=reward,
                               timestamp=self.episode_length * self.skip_frame)

        return task_observations, reward, done, info

    def sample_new_task(self):
        raise Exception(" ")

    def switch_task(self, task):
        self.task = task
        self.task.init_task(self.robot, self.stage)
        selected_observations = self.task.observation_keys
        self.robot.select_observations(selected_observations)
        self.stage.select_observations(selected_observations)
        self.observation_space = \
            combine_spaces(self.robot.get_observation_spaces(),
                           self.stage.get_observation_spaces())
        self.action_space = self.robot.get_action_spaces()

    def get_counterfactual_world(self):
        raise Exception(" ")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self):
        self.episode_length = 0
        if self.logging:
            # TODO: Replace this with another task method in the future providing additional params for the reinit
            self.logger.new_episode(task_params=self.task.get_description())
        return self.task.reset_task()

    def close(self):
        if self.logging:
            self.logger.save('{}.pickle'.format(self.task.id))
        self.robot.close()

    def render(self, mode='human'):
        raise Exception(" ")

    def enforce_max_episode_length(self, episode_length=2000):
        self.enforce_episode_length = True
        self.max_episode_length = episode_length

    def _is_done(self):
        if self.enforce_episode_length and self.episode_length > self.max_episode_length:
            return True
        else:
            return self.task.is_done()

    def do_random_intervention(self):
        self.task.do_random_intervention()

    def get_full_state(self):
        full_state = []
        full_state.extend(self.robot.get_full_state())
        full_state.extend(self.stage.get_full_state())
        return np.array(full_state)

    def set_full_state(self, new_full_state):
        robot_state_size = self.robot.get_state_size()
        self.robot.set_full_state(new_full_state[0:robot_state_size])
        self.stage.set_full_state(new_full_state[robot_state_size:])
        return
