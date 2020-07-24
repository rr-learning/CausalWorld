from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.intervention_actors import GoalInterventionActorPolicy, VisualInterventionActorPolicy, \
    RandomInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    #initialize env
    task = task_generator(task_generator_id='reaching')
    env = CausalWorld(task, skip_frame=10, enable_visualization=True)

    # define a custom curriculum of interventions:
    # Goal intervention actor each episode after reset

    env = CurriculumWrapper(env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])

    for reset_idx in range(30):
        obs = env.reset()
        for time in range(100):
            desired_action = env.action_space.sample()
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    example()
