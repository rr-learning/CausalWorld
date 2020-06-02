from causal_rl_bench.agents.reacher_policy import ReacherActorPolicy
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from causal_rl_bench.intervention_agents import GoalInterventionActorPolicy
from causal_rl_bench.curriculum import InterventionsCurriculum
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper


def example():
    #initialize env
    task_gen = task_generator(task_generator_id='reaching')
    env = World(task_gen, skip_frame=1,
                enable_visualization=True)

    #get policy/controller
    reacher_policy = ReacherActorPolicy()

    def policy_fn(obs):
        return reacher_policy.act(obs)

    #define a curriculum for running and testing it
    #lets get the goal intervention agent and impose it every 3 episodes
    curr_curriculum = InterventionsCurriculum(intervention_actors=[GoalInterventionActorPolicy()],
                                              episodes_hold=[3],
                                              timesteps_hold=[None])
    env = CurriculumWrapper(env,
                            interventions_curriculum=curr_curriculum)

    for reset_idx in range(40):
        obs = env.reset()
        for time in range(500):
            desired_action = policy_fn(obs)
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    example()