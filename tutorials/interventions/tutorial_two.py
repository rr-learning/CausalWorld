from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import numpy as np


def without_intervention_split():
    task = task_generator(task_generator_id='pushing')
    env = CausalWorld(task=task, enable_visualization=True)
    env.reset()
    for _ in range(10):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal = env.do_intervention({'stage_color':
                                                  np.random.uniform(0,
                                                                    1,
                                                                    [3, ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_1():
    task = task_generator(task_generator_id='pushing', intervention_split=True,
                          training=True)
    env = CausalWorld(task=task, enable_visualization=False)
    env.reset()
    for _ in range(10):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal = env.do_intervention({'stage_color': np.random.uniform(0, 1, [3, ])})
        print("Intervention success signal", success_signal)
    env.close()


def with_intervention_split_2():
    task = task_generator(task_generator_id='pushing', intervention_split=True,
                          training=True)
    env = CausalWorld(task=task, enable_visualization=False)
    interventions_space = task.get_training_intervention_spaces()
    env.reset()
    for _ in range(10):
        for i in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
        success_signal = env.do_intervention({'stage_color':
                                                  np.random.uniform(interventions_space['stage_color'][0],
                                                                    interventions_space['stage_color'][1])})
        print("Intervention success signal", success_signal)
    env.close()


if __name__ == '__main__':
    without_intervention_split()
    # with_intervention_split_1()
    # with_intervention_split_2()