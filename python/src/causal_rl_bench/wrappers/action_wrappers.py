from causal_rl_bench.wrappers.policy_wrappers import MovingAverageActionWrapperActorPolicy
from causal_rl_bench.agents.dummy_policy import DummyActorPolicy
import gym


class DeltaActionEnvWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DeltaActionEnvWrapper, self).__init__(env)
        #TODO: discuss the action space of a delta action
        self.env._add_wrapper_info({'delta_action': dict()})

    def action(self, action):
        #Take care of normalization here too
        if self.env.get_action_mode() == "joint_positions":
            #The delta is wrt the last applied
            # joint positions that were sent to the pd controller
            offset = self.env.get_robot().get_last_applied_joint_positions()
        elif self.env.get_action_mode() == "joint_torques":
            offset = self.env.get_robot().get_latest_full_state()['torques']
        elif self.env.get_action_mode() == "end_effector_positions":
            # applied joint positions that were sent to the pd controller
            offset = self.env.get_robot().get_latest_full_state()['end_effector_positions']
        else:
            raise Exception("action mode is not known")
        if self.env.are_actions_normalized():
            offset = self.env.get_robot().normalize_observation_for_key(
                    observation=offset, key=self.env.get_action_mode())
        return action + offset

    def reverse_action(self, action):
        if self.env.get_action_mode() == "joint_positions":
            offset = self.env.get_robot().get_last_applied_joint_positions()
        elif self.env.get_action_mode() == "joint_torques":
            offset = self.env.get_robot().get_latest_full_state()['torques']
        elif self.env.get_action_mode() == "end_effector_positions":
            offset = self.env.get_robot().get_latest_full_state()['end_effector_positions']
        else:
            raise Exception("action mode is not known")
        if self.env.are_actions_normalized():
            offset = self.env.get_robot().normalize_observation_for_key(
                    observation=offset, key=self.env.action_mode)
        return action - offset

    def reset(self, interventions_dict=None):
        return self.env.reset(interventions_dict)


class MovingAverageActionEnvWrapper(gym.ActionWrapper):
    def __init__(self, env, widow_size=8, initial_value=0):
        super(MovingAverageActionEnvWrapper, self).__init__(env)
        self._policy = DummyActorPolicy()
        self._policy = MovingAverageActionWrapperActorPolicy(self._policy,
                                                             widow_size=widow_size,
                                                             initial_value=initial_value)
        self.env._add_wrapper_info({'moving_average_action': {'widow_size': widow_size,
                                                             'initial_value': initial_value}})
        return

    def action(self, action):
        self._policy.policy.add_action(action) #hack now
        return self._policy.act(observation=None)

    def reverse_action(self, action):
        raise Exception("not implemented yet")

    def reset(self, interventions_dict=None):
        return self.env.reset(interventions_dict)
