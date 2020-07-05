from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator


def goal_interventions():
    task = task_generator(task_generator_id='picking')
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