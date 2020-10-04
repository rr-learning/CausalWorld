"""
This tutorial shows you how to solve reaching task using inverse kinemetics
provided in the package internally.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def control_policy(env, obs):
    return \
        env.get_robot().get_joint_positions_from_tip_positions(
            obs[-9:], obs[1:10])


def end_effector_pos():
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      action_mode="joint_positions",
                      normalize_actions=False,
                      normalize_observations=False)
    obs = env.reset()
    for _ in range(100):
        goal_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_dict)
        obs, reward, done, info = env.step(control_policy(env, obs))
        for _ in range(250):
            obs, reward, done, info = env.step(control_policy(env, obs))
    env.close()


if __name__ == '__main__':
    end_effector_pos()
