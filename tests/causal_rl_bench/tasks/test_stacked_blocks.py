from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def test_task_sampler():
    task = Task(task_id='stacked_blocks')
    env = World(task=task, enable_visualization=True, seed=0)
    desired_action = env.action_space.sample
    for i in range(40):
        for _ in range(2):
            obs = env.reset()
            for _ in range(200):
                obs, reward, done, info = env.step(env.action_space.sample())
                print(reward)
            env.do_intervention(variable_name='stack_levels', variable_value=2)
            env.do_random_intervention()
            for _ in range(200):
                obs, reward, done, info = env.step(env.action_space.sample())
                print(reward)
        env.task.generate_new_goal()
    env.close()


test_task_sampler()
