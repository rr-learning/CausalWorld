from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.env_wrappers import HERGoalEnvWrapper
from causal_rl_bench.intervention_agents.training_intervention import \
    reset_training_intervention_agent
from causal_rl_bench.wrappers.intervention_wrappers import \
    ResetInterventionsActorWrapper


def example():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True)
    training_intervention_agent = \
        reset_training_intervention_agent(task_generator_id='picking')
    env = ResetInterventionsActorWrapper(env, training_intervention_agent)
    for _ in range(50):
        obs = env.reset()
        print(env.observation_space)
        # print(obs)
        # chosen_intervention = env.do_single_random_intervention()
        # print(chosen_intervention)
        # print("intervened")
        for _ in range(200):
            obs, reward, done, info = env.step(env.action_space.sample())
            # print(obs[-9:])
            # print(reward)
            # print(done)
            # print(info)
    # env.save_world('./configs')
    env.close()

    # from causal_rl_bench.utils.config_utils import load_world
    # env = load_world('./configs', enable_visualization=True)
    # for _ in range(3):
    #     obs = env.reset()
    #     new_goal = env.sample_new_goal()
    #     print(new_goal)
    #     # print(obs)
    #     # chosen_intervention = env.do_single_random_intervention()
    #     # print(chosen_intervention)
    #     # print("intervened")
    #     for _ in range(1000):
    #         obs, reward, done, info = env.step(env.action_space.sample())
    #         # print(obs[-9:])
    #         # print(reward)
    #         # print(done)
    #         # print(info)



if __name__ == '__main__':
    example()
