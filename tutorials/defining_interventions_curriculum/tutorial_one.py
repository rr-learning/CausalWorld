from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.policy_wrapper import PolicyWrapper
from stable_baselines import PPO2
import causal_rl_bench.viewers.task_viewer as viewer
from causal_rl_bench.envs.world import World
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.curriculum.interventions_curriculum import InterventionsCurriculumWrapper
from causal_rl_bench.intervention_policies.random import StudentRandomInterventionPolicy, \
    TeacherRandomInterventionPolicy


def without_interventions():
    task = Task(task_id='reaching')
    stable_baselines_policy_path = "./saved_model.zip"
    model = PPO2.load(stable_baselines_policy_path)

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

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
    stable_baselines_policy_path = "./saved_model.zip"
    model = PPO2.load(stable_baselines_policy_path)

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    env = World(task, skip_frame=1,
                enable_visualization=True)

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
                                         teacher_episode_hold=10)
    for reset_idx in range(40):
        obs = env.reset()
        for time in range(960):
            desired_action = policy_fn(obs)
            obs, reward, done, info = env.step(action=desired_action)
    env.close()


if __name__ == '__main__':
    with_interventions()
