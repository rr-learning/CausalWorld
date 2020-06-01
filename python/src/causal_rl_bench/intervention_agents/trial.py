import numpy as np
from causal_rl_bench.intervention_agents.base_policy import \
    BaseInterventionActorPolicy
from causal_rl_bench.utils.rotation_utils import quaternion_to_euler, \
    euler_to_quaternion


class TrialInterventionActorPolicy(BaseInterventionActorPolicy):
    def __init__(self):
        super(TrialInterventionActorPolicy, self).__init__()
        self.task_intervention_space = None
        self.low_joint_positions = None

    def initialize_actor(self, env):
        self.task_intervention_space =\
            env.task.get_testing_intervention_spaces()
        self.task_intervention_space.\
            update(env.task.get_training_intervention_spaces())
        self.low_joint_positions = env.robot.\
            robot_actions.joint_positions_lower_bounds
        return

    def _act(self, variables_dict):
        interventions_dict = dict()
        interventions_dict['joint_positions'] = self.low_joint_positions
        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict) \
                    and variable.startswith("tool"):
                interventions_dict[variable] = dict()
                # interventions_dict[variable]['position'] = \
                #     variables_dict[variable]['position']
                # interventions_dict[variable]['position'][-1] += 0.002
                euler_orientation = \
                    quaternion_to_euler(variables_dict
                                        [variable]['orientation'])
                euler_orientation[0] += 0.1
                interventions_dict[variable]['orientation'] = \
                    euler_to_quaternion(euler_orientation)
        return interventions_dict

    def get_intervention_actor_params(self):
        #TODO: We need to think about how to save its params more and load them?
        #potentially?
        intervention_params = dict()
        intervention_params["intervention_actor_name"] = \
            "rigid_block_position"
        return intervention_params
