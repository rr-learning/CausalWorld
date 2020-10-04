"""
This tutorial shows you how to use privileged information to solve the
task at hand.
"""
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task


def privileged_information():
    task = generate_task(task_generator_id='general')
    env = CausalWorld(task=task, enable_visualization=True)
    env.expose_potential_partial_solution()
    env.reset()
    for _ in range(10):
        goal_intervention_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.low)
        print("now we solve it with privileged info")
        success_signal, obs = env.do_intervention(
            info['possible_solution_intervention'], check_bounds=False)
        print("Partial Solution Setting Intervention Succes Signal",
              success_signal)
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.low)
        print("fractional_success is:", info['fractional_success'])
    env.close()


if __name__ == '__main__':
    privileged_information()
