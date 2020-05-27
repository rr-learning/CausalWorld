from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task


def example():
    task = Task(task_id='reaching')
    env = World(task=task, skip_frame=1, enable_visualization=True, seed=0)
    env.reset()
    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
    #now lets try to get the bounds I am allowed to intervene on
    print(task.get_training_intervention_spaces())
    success_signal = env.do_intervention(variable_name='goal_positions',
                                         variable_value=[0, 0.2,  0.1, 0, 0.1,
                                                         0.15, 0, 0.1,  0.25])
    print(success_signal)

    success_signal = env.do_intervention(variable_name='goal_positions',
                                         variable_value=[-0.1, -0.05, 0.1,
                                                         -0.1, -0.05, 0.15,
                                                         -0.1, -0.05, 0.05])
    print(success_signal)

    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    example()
