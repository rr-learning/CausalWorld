from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy


class ReacherInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self, **kwargs):
        super(ReacherInterventionActorPolicy, self).__init__()
        self.joint_position_sampler_func = None
        self.goal_position_sampler_func = None

    def initialize(self, env):
        self.joint_position_sampler_func = \
            env.robot.sample_joint_positions
        self.goal_position_sampler_func = \
            env.robot.sample_end_effector_positions
        return

    def _act(self, variables_dict):
        interventions_dict = dict()
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

    def get_params(self):
        return {'reacher_agent': dict()}

