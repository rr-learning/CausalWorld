import gym
import numpy as np


class DeltaAction(gym.ActionWrapper):
    def __init__(self, env):
        super(DeltaAction, self).__init__(env)
        #TODO: discuss the action space of a delta action

    def action(self, action):
        #Take care of normalization here too
        if self.env.action_mode == "joint_positions":
            #The delta is wrt the last applied
            # joint positions that were sent to the pd controller
            offset = self.env.robot.last_applied_joint_positions
        elif self.env.action_mode == "joint_torques":
            offset = self.env.robot.latest_full_state.torques
        elif self.env.action_mode == "end_effector_positions":
            # applied joint positions that were sent to the pd controller
            offset = self.env.robot.compute_end_effector_positions(
                self.env.robot.last_applied_joint_positions)
        else:
            raise Exception("action mode is not known")
        if self.env.robot.normalize_actions:
            offset = self.env.robot.normalize_observation_for_key(
                    observation=offset, key=self.env.action_mode)
        return action + offset

    def reverse_action(self, action):
        if self.env.action_mode == "joint_positions":
            offset = self.env.robot.last_applied_joint_positions
        elif self.env.action_mode == "joint_torques":
            offset = self.env.robot.latest_full_state.torques
        elif self.env.action_mode == "end_effector_positions":
            offset = self.env.robot.compute_end_effector_positions(
                         self.env.robot.last_applied_joint_positions)
        else:
            raise Exception("action mode is not known")
        if self.env.robot.normalize_actions:
            offset = self.env.robot.normalize_observation_for_key(
                    observation=offset, key=self.env.action_mode)
        return action - offset
