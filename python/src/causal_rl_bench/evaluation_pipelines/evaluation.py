from causal_rl_bench.tasks.task import Task
from causal_rl_bench.envs.world import World
from causal_rl_bench.metrics.mean_sucess_rate_metric import MeanSucessRateMetric
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.curriculum.interventions_curriculum import InterventionsCurriculumWrapper
from causal_rl_bench.intervention_policies.random import StudentRandomInterventionPolicy, \
    TeacherRandomInterventionPolicy
import numpy as np


class EvaluationPipeline:
    def __init__(self, policy, tracker=None,
                 world_params=None, task_params=None,
                 output_path=None, seed=0, num_seeds=50, runs_per_seed=100):
        # For now we just assume that there were no interventions during training
        # and all variables are held constant throughout training
        # Later when interventions are allowed during training, this will get more
        # complex as it is relevant which variable values have been seen in training
        self.policy_fn = policy
        self.seed = seed
        self.num_seeds = num_seeds
        self.tracker = tracker
        if output_path is None:
            self.output_path = None
        else:
            self.output_path = output_path
        self.seed = 0
        self.data_recorder = DataRecorder(output_directory=None)
        if self.tracker:
            task_stats = self.tracker.task_stats_log[0]
            self.task = Task(task_id=task_stats.task_name, **task_stats.task_params)
        else:
            self.task = Task(**task_params)
        if self.tracker:
            self.env = World(self.task,
                             **self.tracker.world_params,
                             seed=self.seed,
                             data_recorder=self.data_recorder)
        else:
            if world_params is not None:
                self.env = World(self.task,
                                 **world_params,
                                 seed=self.seed,
                                 data_recorder=self.data_recorder)
            else:
                self.env = World(self.task,
                                 seed=self.seed,
                                 data_recorder=self.data_recorder)
        self.evaluation_episode_length_in_secs = 1
        self.time_steps_for_evaluation = int(self.evaluation_episode_length_in_secs / self.env.robot.dt)
        self.metrics_list = []
        self.metrics_list.append(MeanSucessRateMetric())
        return

    def run_episode(self):
        obs = self.env.reset()
        for _ in range(self.time_steps_for_evaluation):
            desired_action = self.policy_fn(obs)
            obs, rew, done, info = self.env.step(desired_action)
        return self.data_recorder.get_current_episode()

    def process_metrics(self, episode):
        for metric in self.metrics_list:
            metric.process_episode(episode)
        return

    def get_metric_scores(self):
        metrics = dict()
        for metric in self.metrics_list:
            metrics[metric.name] = metric.get_metric_score()
        return metrics

    def evaluate_reacher_interventions_curriculum(self, num_of_episodes=100):
        student_intervention_policy = \
            StudentRandomInterventionPolicy(
                self.task.get_testing_intervention_spaces())
        teacher_intervention_policy = \
            TeacherRandomInterventionPolicy(
                self.task.get_testing_intervention_spaces())
        teacher_intervention_policy.initialize_sampler(self.env.robot.
                                                       sample_end_effector_positions)
        student_intervention_policy.initialize_sampler(self.env.robot.
                                                       sample_joint_positions)
        self.env = InterventionsCurriculumWrapper(env=self.env,
                                                  student_policy=
                                                  student_intervention_policy,
                                                  student_episode_hold=2,
                                                  teacher_policy=
                                                  teacher_intervention_policy,
                                                  teacher_episode_hold=4)
        for _ in range(num_of_episodes):
            current_episode = self.run_episode()
            self.process_metrics(current_episode)
        self.env.close()
        scores = self.get_metric_scores()
        scores['total_intervention_steps'] = self.env.tracker.get_total_intervention_steps()
        scores['total_interventions'] = self.env.tracker.get_total_interventions()
        scores['total_timesteps'] = self.env.tracker.get_total_time_steps()
        scores['total_resets'] = self.env.tracker.get_total_resets()
        return scores

    # def evaluate_generalisation(self):
    #     visual_variables = task.get_visual_variables()
    #     scores = dict()
    #     score = dict()
    #     for key in visual_variables.keys():
    #         scores_var_sweep = dict()
    #         for value in visual_variables[key]:
    #             rews = []
    #             dones = 0
    #             for episode_seed in range(self.num_seeds):
    #                 env.do_intervention(**{key: value})
    #                 env.seed(episode_seed)
    #                 obs = env.reset()
    #                 accumulated_reward = 0
    #                 for _ in range(self.tracker.world_params["max_episode_length"]):
    #                     obs, rew, done, info = env.step(self.policy_fn(obs))
    #                     accumulated_reward += rew
    #                     if done:
    #                         dones += 1
    #                         break
    #                 rews.append(accumulated_reward)
    #             score["mean_success"] = float(dones / self.num_seeds)
    #             score["mean_reward"] = np.mean(rews)
    #             score["std_reward"] = np.std(rews)
    #             scores_var_sweep[value] = score
    #         scores[key] = scores_var_sweep
    #     return scores

    def evaluate_teacher_interventions(self):
        pass

    def evaluate_master_interventions(self):
        pass

    def evaluate_runtime_interventions(self):
        pass

    def evaluate_confounding_robustness(self):
        pass

    def run_full_evaluation(self):
        pass
