from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator


def privileged_information():
    task = task_generator(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(10):
        goal_intervention_dict = env.sample_new_goal()
        success_signal = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)
        for i in range(1000):
            obs, reward, done, info = env.step(env.action_space.low)
        print("now we solve it with privileged info")
        success_signal = env.do_intervention(info['possible_solution_intervention'])
        print("Partial Solution Setting Intervention Succes Signal", success_signal)
        for i in range(500):
            obs, reward, done, info = env.step(env.action_space.low)
            print(reward)
    env.close()


if __name__ == '__main__':
    privileged_information()