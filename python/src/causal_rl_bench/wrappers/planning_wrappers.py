from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy
from causal_rl_bench.utils.rotation_utils import quaternion_to_euler, \
    euler_to_quaternion
import gym
import numpy as np


class ObjectSelectorActorPolicy(BaseInterventionActorPolicy):
    def __init__(self):
        super(ObjectSelectorActorPolicy, self).__init__()
        self.low_joint_positions = None
        self.current_action = None
        self.selected_object = None

    def initialize_actor(self, env):
        self.low_joint_positions = env.robot.\
            robot_actions.joint_positions_lower_bounds
        return

    def add_action(self, action, selected_object):
        self.current_action = action
        self.selected_object = selected_object

    def _act(self, variables_dict):
        # action 1 is [0, 1, 2, 3, 4, 5, 6] 0 do nothing, 1 go up 2 down, 3 right, 4 left, 5, front, 6 back
        # action 2 is [0, 1, 2, 3, 4, 5, 6] 0 do nothing, 1 yaw clockwise, 2 yaw anticlockwise, 3 roll clockwise,
        # 4 roll anticlockwise, 5 pitch clockwise, 6 pitch anticlockwise
        interventions_dict = dict()
        interventions_dict['joint_positions'] = self.low_joint_positions
        interventions_dict[self.selected_object] = dict()
        if self.current_action[1] != 0:
            interventions_dict[self.selected_object]['position'] = \
                variables_dict[self.selected_object]['position']
            if self.current_action[1] == 1:
                interventions_dict[self.selected_object]['position'][-1] += 0.002
            elif self.current_action[1] == 2:
                interventions_dict[self.selected_object]['position'][-1] -= 0.002
            elif self.current_action[1] == 3:
                interventions_dict[self.selected_object]['position'][0] += 0.002
            elif self.current_action[1] == 4:
                interventions_dict[self.selected_object]['position'][0] -= 0.002
            elif self.current_action[1] == 5:
                interventions_dict[self.selected_object]['position'][1] += 0.002
            elif self.current_action[1] == 6:
                interventions_dict[self.selected_object]['position'][1] -= 0.002
            else:
                print(self.current_action[1])
                raise Exception("The passed action mode is not supported")
        if self.current_action[2] != 0:
            euler_orientation = \
                quaternion_to_euler(variables_dict
                                    [self.selected_object]['orientation'])
            if self.current_action[2] == 1:
                euler_orientation[-1] += 0.1
            elif self.current_action[2] == 2:
                euler_orientation[-1] -= 0.1
            elif self.current_action[2] == 3:
                euler_orientation[1] += 0.1
            elif self.current_action[2] == 4:
                euler_orientation[1] -= 0.1
            elif self.current_action[2] == 5:
                euler_orientation[0] += 0.1
            elif self.current_action[2] == 6:
                euler_orientation[0] -= 0.1
            else:
                print(self.current_action[2])
                raise Exception("The passed action mode is not supported")
            interventions_dict[self.selected_object]['orientation'] = \
                euler_to_quaternion(euler_orientation)
        return interventions_dict


class ObjectSelectorWrapper(gym.Wrapper):
    #TODO: we dont support changing observation space size for this wrapper now
    #TODO: we dont support normalized observation space for now
    def __init__(self, env):
        super(ObjectSelectorWrapper, self).__init__(env)
        self.env = env
        self.intervention_actor = ObjectSelectorActorPolicy()
        self.intervention_actor.initialize_actor(self.env)
        self.env._disable_actions()
        self.observations_order = []
        self.observation_high = []
        curr_variables = self.env.get_current_task_parameters()
        for intervention_variable in curr_variables:
            if intervention_variable.startswith("tool"):
                self.observations_order.append(intervention_variable)
                self.observation_high.append(np.repeat(np.finfo(np.float32).max, 9))
        self.observation_high = np.array(self.observation_high)
        self.observation_space = gym.spaces.Box(-self.observation_high, self.observation_high)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(len(self.observations_order)),
                                              gym.spaces.Discrete(7),
                                              gym.spaces.Discrete(7)))
        self.observations_order.sort()
        self.env._add_wrapper_info({'object_selector': dict()})

    def step(self, action):
        #buffer action to the intervention actor
        self.intervention_actor.add_action(action, self.observations_order[action[0]])
        intervention_dict = self.intervention_actor.act(
            self.env.get_current_task_parameters())
        self.env.do_intervention(intervention_dict)
        obs, reward, done, info = self.env.step(self.env.action_space.low)
        #we will keep done, reward and info and change the observation space
        curr_variables = self.env.get_current_task_parameters()
        new_observations = []
        for observation_var in self.observations_order:
            new_observations.append(curr_variables[observation_var]['position'])
            new_observations.append(quaternion_to_euler(curr_variables[observation_var]['orientation']))
            new_observations.append(curr_variables[observation_var]['size'])
        return np.array(new_observations).flatten(), reward, done, info
