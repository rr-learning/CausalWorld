from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.agents.reacher_policy import ReacherActorPolicy
from causal_rl_bench.wrappers.action_wrappers import \
    MovingAverageActionEnvWrapper
from causal_rl_bench.wrappers.policy_wrappers import \
    MovingAverageActionWrapperActorPolicy
from causal_rl_bench.curriculum.interventions_curriculum \
    import InterventionsCurriculumWrapper
from causal_rl_bench.meta_agents.random import RandomMetaActorPolicy


def without_interventions():
    task = Task(task_generator_id='reaching')
    reacher_policy = ReacherActorPolicy()

    def policy_fn(obs):
        return reacher_policy.act(obs)

    env = World(task, skip_frame=1,
                enable_visualization=True)
    for reset_idx in range(40):
        obs = env.reset()
        for time in range(960):
            desired_action = policy_fn(obs)
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


def with_interventions():
    task = Task(task_generator_id='reaching')
    reacher_policy = ReacherActorPolicy()
    # reacher_policy = MovingAverageActionPolicyWrapper(reacher_policy,
    #                                                   widow_size=250)

    def policy_fn(obs):
        return reacher_policy.act(obs)

    env = World(task, skip_frame=1,
                enable_visualization=True)
    env = MovingAverageActionEnvWrapper(env, widow_size=250)

    #till here its the same step, now we need a curriculum wrapper with the intervention agents
    #lets try only a student intervention
    meta_actor_policy = RandomMetaActorPolicy(
        task.get_testing_intervention_spaces())
    meta_actor_policy.add_sampler_func(variable_name='goal_positions',
                                       sampler_func=env.robot.
                                       sample_end_effector_positions)
    meta_actor_policy.add_sampler_func(variable_name='joint_positions',
                                       sampler_func=env.robot.
                                       sample_joint_positions)
    env = InterventionsCurriculumWrapper(env=env,
                                         meta_actor_policy=
                                         meta_actor_policy,
                                         meta_episode_hold=4)

    for reset_idx in range(40):
        obs = env.reset()
        for time in range(960):
            desired_action = policy_fn(obs)
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    with_interventions()
