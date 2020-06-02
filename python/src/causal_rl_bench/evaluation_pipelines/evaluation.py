import os
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from causal_rl_bench.metrics.mean_sucess_rate_metric import \
    MeanSuccessRateMetric
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.wrappers.curriculum_wrappers \
    import CurriculumWrapper
from causal_rl_bench.loggers.tracker import Tracker


class EvaluationPipeline(object):
    def __init__(self, testing_curriculum,
                 tracker_path=None, world_params=None,
                 task_params=None, intervention_split=True,
                 training=False, visualize_evaluation=False,
                 initial_seed=0):
        self.intervention_split = intervention_split
        self.training = training
        self.initial_seed = initial_seed
        self.testing_curriculum = testing_curriculum
        self.data_recorder = DataRecorder(output_directory=None)
        if tracker_path is not None:
            self.tracker = Tracker(
                file_path=os.path.join(tracker_path, 'tracker'))
            task_stats = self.tracker.task_stats_log[0]
            del task_stats.task_params['intervention_split']
            del task_stats.task_params['training']
            self.task = task_generator(task_generator_id=task_stats.task_name,
                                       **task_stats.task_params,
                                       intervention_split=intervention_split,
                                       training=training)
        else:
            if 'intervention_split' in task_params or 'training' in task_params:
                del task_params['intervention_split']
                del task_params['training']
            self.task = task_generator(**task_params,
                                       intervention_split=intervention_split,
                                       training=training)
        if tracker_path:
            if world_params is not None:
                if 'seed' in self.tracker.world_params:
                    del self.tracker.world_params['seed']
            self.env = World(self.task,
                             **self.tracker.world_params,
                             seed=self.initial_seed,
                             data_recorder=self.data_recorder,
                             enable_visualization=visualize_evaluation)
        else:
            if world_params is not None:
                if 'seed' in world_params:
                    del world_params['seed']
                self.env = World(self.task,
                                 **world_params,
                                 seed=self.initial_seed,
                                 data_recorder=self.data_recorder,
                                 enable_visualization=visualize_evaluation)
            else:
                self.env = World(self.task,
                                 seed=self.initial_seed,
                                 data_recorder=self.data_recorder,
                                 enable_visualization=visualize_evaluation)
        evaluation_episode_length_in_secs = 1
        self.time_steps_for_evaluation = \
            int(evaluation_episode_length_in_secs / self.env.robot.dt)
        #wrap now the environment
        self.env = CurriculumWrapper(self.env,
                                     interventions_curriculum=self.testing_curriculum)

        self.metrics_list = []
        self.metrics_list.append(MeanSuccessRateMetric())
        return

    def run_episode(self, policy_fn):
        obs = self.env.reset()
        for _ in range(self.time_steps_for_evaluation):
            desired_action = policy_fn(obs)
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

    def evaluate_policy(self, policy, num_seeds=5,
                        episodes_per_seed=5):
        evaluation_budget = self.time_steps_for_evaluation * \
                            episodes_per_seed * num_seeds
        for i in range(num_seeds):
            self.env.seed(seed=self.initial_seed + i)
            self.testing_curriculum.reset()
            for _ in range(episodes_per_seed):
                current_episode = self.run_episode(policy)
                self.process_metrics(current_episode)
                self.data_recorder.clear_recorder()
        self.env.close()
        scores = self.get_metric_scores()
        scores['total_intervention_steps'] = \
            self.env.tracker.get_total_intervention_steps()
        scores['total_interventions'] = \
            self.env.tracker.get_total_interventions()
        scores['total_timesteps'] = \
            self.env.tracker.get_total_time_steps()
        scores['num_of_seeds'] = \
            num_seeds
        scores['episodes_per_seed'] = \
            episodes_per_seed
        scores['limited_exposed_intervention_variables'] = \
            self.intervention_split
        if self.intervention_split:
            if self.training:
                scores['intervention_set_chosen'] = \
                    "training"
            else:
                scores['intervention_set_chosen'] = \
                    "testing"
        scores['total_resets'] = \
            self.env.tracker.get_total_resets()
        scores['total_invalid_intervention_steps'] = \
            self.env.tracker.get_total_invalid_intervention_steps()
        scores['total_invalid_robot_intervention_steps'] = \
            self.env.tracker.get_total_invalid_robot_intervention_steps()
        scores['total_invalid_stage_intervention_steps'] = \
            self.env.tracker.get_total_invalid_stage_intervention_steps()
        scores['total_invalid_task_generator_intervention_steps'] = \
            self.env.tracker.get_total_invalid_task_generator_intervention_steps()
        scores['total_invalid_out_of_bounds_intervention_steps'] = \
            self.env.tracker.get_total_invalid_out_of_bounds_intervention_steps()
        return scores
