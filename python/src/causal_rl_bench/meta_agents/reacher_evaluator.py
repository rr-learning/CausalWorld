from causal_rl_bench.meta_agents.base_policy import BaseMetaActorPolicy
import numpy as np


class ReacherMetaActorPolicy(BaseMetaActorPolicy):
    def __init__(self, joint_position_sampler_func,
                 goal_position_sampler_func):
        super(ReacherMetaActorPolicy, self).__init__()
        self.joint_position_sampler_func = joint_position_sampler_func
        self.goal_position_sampler_func = goal_position_sampler_func

    def _act(self, variables_dict):
        #TODO: double check that u dont intervene on a variable that
        # u dont get as input
        interventions_dict = dict()
        #have static interventions here basically
        interventions_dict['joint_positions'] = \
            self.joint_position_sampler_func()
        new_goal = self.goal_position_sampler_func()
        interventions_dict['goal_60'] = dict()
        interventions_dict['goal_60']['position'] = new_goal[:3]
        interventions_dict['goal_120'] = dict()
        interventions_dict['goal_120']['position'] = new_goal[3:6]
        interventions_dict['goal_300'] = dict()
        interventions_dict['goal_300']['position'] = new_goal[6:]
        new_floor_color = np.array(variables_dict['floor_color'])
        new_floor_color[-1] += 0.1
        interventions_dict['floor_color'] = new_floor_color
        return interventions_dict
