"""
This tutorial shows you how to sample new goals by intervening
on the environment.
"""
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def goal_interventions():
    task = generate_task(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(10):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        goal_intervention_dict = env.sample_new_goal()
        print("new goal chosen: ", goal_intervention_dict)
        success_signal, obs = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)
    env.close()


if __name__ == '__main__':
    goal_interventions()