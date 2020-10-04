from causal_world.wrappers.policy_wrappers import \
    MovingAverageActionWrapperActorPolicy
from causal_world.actors.dummy_policy import DummyActorPolicy
import gym


class DeltaActionEnvWrapper(gym.ActionWrapper):

    def __init__(self, env):
        """
        A delta action wrapper for the environment to turn the actions
        to a delta wrt the previous action executed.

        :param env: (causal_world.CausalWorld) the environment to convert.
        """
        super(DeltaActionEnvWrapper, self).__init__(env)
        self.env.add_wrapper_info({'delta_action': dict()})

    def action(self, action):
        """
        Processes the action to transform to a delta action.

        :param action: (nd.array) the raw action to be processed.
        :return: (nd.array) the delta action.
        """
        if self.env.get_action_mode() == "joint_positions":
            offset = self.env.get_robot().get_last_applied_joint_positions()
        elif self.env.get_action_mode() == "joint_torques":
            offset = self.env.get_robot().get_latest_full_state()['torques']
        elif self.env.get_action_mode() == "end_effector_positions":
            offset = self.env.get_robot().get_latest_full_state(
            )['end_effector_positions']
        else:
            raise Exception("action mode is not known")
        if self.env.are_actions_normalized():
            offset = self.env.get_robot().normalize_observation_for_key(
                observation=offset, key=self.env.get_action_mode())
        return action + offset

    def reverse_action(self, action):
        """
        Reverses processing the action to transform to a raw action again.

        :param action: (nd.array) the delta action.
        :return: (nd.array) the raw action before processing.
        """
        if self.env.get_action_mode() == "joint_positions":
            offset = self.env.get_robot().get_last_applied_joint_positions()
        elif self.env.get_action_mode() == "joint_torques":
            offset = self.env.get_robot().get_latest_full_state()['torques']
        elif self.env.get_action_mode() == "end_effector_positions":
            offset = self.env.get_robot().get_latest_full_state(
            )['end_effector_positions']
        else:
            raise Exception("action mode is not known")
        if self.env.are_actions_normalized():
            offset = self.env.get_robot().normalize_observation_for_key(
                observation=offset, key=self.env.action_mode)
        return action - offset


class MovingAverageActionEnvWrapper(gym.ActionWrapper):

    def __init__(self, env, widow_size=8, initial_value=0):
        """

        :param env: (causal_world.CausalWorld) the environment to convert.
        :param widow_size: (int) the window size for avergaing and smoothing
                                 the actions.
        :param initial_value: (float) intial values to fill the window with.
        """
        super(MovingAverageActionEnvWrapper, self).__init__(env)
        self._policy = DummyActorPolicy()
        self._policy = MovingAverageActionWrapperActorPolicy(
            self._policy, widow_size=widow_size, initial_value=initial_value)
        self.env.add_wrapper_info({
            'moving_average_action': {
                'widow_size': widow_size,
                'initial_value': initial_value
            }
        })
        return

    def action(self, action):
        """
        Processes the action to transform to a smoothed action.

        :param action: (nd.array) the raw action to be processed.
        :return: (nd.array) the smoothed action.
        """
        self._policy.policy.add_action(action)
        return self._policy.act(obs=None)

    def reverse_action(self, action):
        """
        Reverses processing the action to transform to a raw action again.

        :param action: (nd.array) the smoothed action.
        :return: (nd.array) the raw action before processing.
        """
        raise Exception("not implemented yet")
