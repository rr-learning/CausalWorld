"""
This tutorial shows you how to use privileged information to solve the
task at hand.
"""
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def privileged_information():
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task, enable_visualization=True, normalize_actions=False)
    env.expose_potential_partial_solution()
    env.reset()
    for _ in range(10):
        goal_intervention_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)
        obs, reward, done, info = env.step(env.action_space.low)
        for i in range(1000):
            obs, reward, done, info = env.step(info['possible_solution_intervention']['joint_positions'])
        print("now we solve it with privileged info")
        print(info['possible_solution_intervention'])
        print("Partial Solution Setting Intervention Succes Signal",
              success_signal)
    env.close()


if __name__ == '__main__':
    privileged_information()
