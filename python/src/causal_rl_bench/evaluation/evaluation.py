import os
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from causal_rl_bench.metrics.mean_sucess_rate_metric import \
    MeanSuccessRateMetric
from causal_rl_bench.envs.world import World
from causal_rl_bench.metrics.mean_accumulated_reward_metric import \
    MeanAccumulatedRewardMetric
from causal_rl_bench.loggers.data_recorder import DataRecorder
from causal_rl_bench.wrappers.protocol_wrapper \
    import ProtocolWrapper
from causal_rl_bench.loggers.tracker import Tracker
import json

class EvaluationPipeline(object):
    def __init__(self, evaluation_protocols, tracker_path=None,
                 world_params=None, task_params=None,
                 intervention_split=True, training=False,
                 visualize_evaluation=False, initial_seed=0):
        self.intervention_split = intervention_split
        self.training = training
        self.initial_seed = initial_seed
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
        evaluation_episode_length_in_secs = self.task.get_max_episode_length()
        self.time_steps_for_evaluation = \
            int(evaluation_episode_length_in_secs / self.env.dt)

        self.evaluation_env = self.env
        self.evaluation_protocols = evaluation_protocols
        self.metrics_list = []
        self.metrics_list.append(MeanSuccessRateMetric())
        self.metrics_list.append(MeanAccumulatedRewardMetric())
        return

    def run_episode(self, policy_fn):
        obs = self.evaluation_env.reset()
        for _ in range(self.time_steps_for_evaluation):
            desired_action = policy_fn(obs)
            obs, rew, done, info = self.evaluation_env.step(desired_action)
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

    def evaluate_policy(self, policy):
        pipeline_scores = dict()
        for evaluation_protocol in self.evaluation_protocols:
            self.evaluation_env = ProtocolWrapper(self.env, evaluation_protocol)
            evaluation_protocol.init(self.evaluation_env, self.env.get_tracker())
            episodes_in_protocol = evaluation_protocol.get_num_episodes()
            for _ in range(episodes_in_protocol):
                current_episode = self.run_episode(policy)
                self.process_metrics(current_episode)
                self.data_recorder.clear_recorder()
            scores = self.get_metric_scores()
            scores['total_intervention_steps'] = \
                self.env.get_tracker().get_total_intervention_steps()
            scores['total_interventions'] = \
                self.env.get_tracker().get_total_interventions()
            scores['total_timesteps'] = \
                self.env.get_tracker().get_total_time_steps()
            # Not sure why we should care about intervention_split during evaluation. That is typically
            # on the designer of the protocol right?
            # scores['limited_exposed_intervention_variables'] = \
            #     self.intervention_split
            # if self.intervention_split:
            #     if self.training:
            #         scores['intervention_set_chosen'] = \
            #             "training"
            #     else:
            #         scores['intervention_set_chosen'] = \
            #             "testing"
            scores['total_resets'] = \
                self.env.get_tracker().get_total_resets()
            # Why are we interested in the invalid interactions here? if they are invalid,
            # that's the protocol designers fault

            # scores['total_invalid_intervention_steps'] = \
            #     self.env.get_tracker().get_total_invalid_intervention_steps()
            # scores['total_invalid_robot_intervention_steps'] = \
            #     self.env.get_tracker().get_total_invalid_robot_intervention_steps()
            # scores['total_invalid_stage_intervention_steps'] = \
            #     self.env.get_tracker().get_total_invalid_stage_intervention_steps()
            # scores['total_invalid_task_generator_intervention_steps'] = \
            #     self.env.get_tracker().get_total_invalid_task_generator_intervention_steps()
            # scores['total_invalid_out_of_bounds_intervention_steps'] = \
            #     self.env.get_tracker().get_total_invalid_out_of_bounds_intervention_steps()
            pipeline_scores[evaluation_protocol.get_name()] = scores
        self.evaluation_env.close()
        self.pipeline_scores = pipeline_scores
        return pipeline_scores

    def save_scores(self, evaluation_path):
        file_path = os.path.join(evaluation_path, 'scores.json')
        with open(file_path, "w") as json_file:
            json.dump(self.pipeline_scores, json_file)

