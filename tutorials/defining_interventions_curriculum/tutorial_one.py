from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.agents.reacher_policy import ReacherPolicy
from causal_rl_bench.wrappers.action_wrappers import MovingAverageActionEnvWrapper
from causal_rl_bench.wrappers.policy_wrappers import MovingAverageActionPolicyWrapper
from causal_rl_bench.curriculum.interventions_curriculum import InterventionsCurriculumWrapper
from causal_rl_bench.intervention_policies.random import StudentRandomInterventionPolicy, \
    TeacherRandomInterventionPolicy


def without_interventions():
    task = Task(task_id='reaching')
    reacher_policy = ReacherPolicy()

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
    task = Task(task_id='reaching')
    reacher_policy = ReacherPolicy()
    # reacher_policy = MovingAverageActionPolicyWrapper(reacher_policy,
    #                                                   widow_size=250)

    def policy_fn(obs):
        return reacher_policy.act(obs)

    env = World(task, skip_frame=1,
                enable_visualization=True)
    env = MovingAverageActionEnvWrapper(env, widow_size=250)

    #till here its the same step, now we need a curriculum wrapper with the intervention agents
    #lets try only a student intervention
    student_intervention_policy = \
        StudentRandomInterventionPolicy(
            task.get_training_intervention_spaces())
    teacher_intervention_policy = \
        TeacherRandomInterventionPolicy(
            task.get_training_intervention_spaces())
    #the below two steps will not be needed if u have a policy or so
    teacher_intervention_policy.initialize_sampler(env.robot.
                                                   sample_end_effector_positions)
    student_intervention_policy.initialize_sampler(env.robot.
                                                   sample_joint_positions)
    #now define the curriculum
    env = InterventionsCurriculumWrapper(env=env,
                                         student_policy=
                                         student_intervention_policy,
                                         student_episode_hold=5,
                                         teacher_policy=
                                         teacher_intervention_policy,
                                         teacher_episode_hold=1)
    for reset_idx in range(40):
        obs = env.reset()
        for time in range(960):
            desired_action = policy_fn(obs)
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    with_interventions()
