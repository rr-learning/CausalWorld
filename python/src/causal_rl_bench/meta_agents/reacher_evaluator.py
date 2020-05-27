from causal_rl_bench.meta_agents.base_policy import BaseMetaActorPolicy


class ReacherMetaActorPolicy(BaseMetaActorPolicy):
    def __init__(self, joint_position_sampler_func,
                 goal_position_sampler_func):
        super(ReacherMetaActorPolicy, self).__init__()
        self.joint_position_sampler_func = joint_position_sampler_func
        self.goal_position_sampler_func = goal_position_sampler_func

    def act(self, variables_dict):
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
        return interventions_dict
