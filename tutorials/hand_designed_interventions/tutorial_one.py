from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def example():
    task = Task(task_generator_id='reaching')
    env = World(task=task, skip_frame=1, enable_visualization=True, seed=0)
    env.reset()
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    #now maybe u wanna intervene a random single intervention, using training intervention space
    chosen_intervention = env.do_random_intervention()
    print(chosen_intervention)
    for _ in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
    #now maybe u wanna intervene a random single intervention, using training intervention space
    chosen_intervention = env.do_random_intervention()
    print(chosen_intervention)
    env.close()


if __name__ == '__main__':
    example()
