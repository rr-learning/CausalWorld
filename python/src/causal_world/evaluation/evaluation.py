import os
from causal_world.task_generators.task import task_generator
from causal_world.metrics.mean_complete_success_rate import \
    MeanCompleteSuccessRate
from causal_world.envs.causalworld import CausalWorld

from causal_world.metrics.mean_last_fractional_success import \
    MeanLastFractionalSuccess
from causal_world.metrics.mean_full_integrated_fractional_success import \
    MeanFullIntegratedFractionalSuccess
from causal_world.metrics.mean_last_integrated_fractional_success import \
    MeanLastIntegratedFractionalSuccess

from causal_world.loggers.data_recorder import DataRecorder
from causal_world.wrappers.protocol_wrapper \
    import ProtocolWrapper
from causal_world.loggers.tracker import Tracker
import json
import logging


class EvaluationPipeline(object):

    def __init__(self,
                 evaluation_protocols,
                 tracker_path=None,
                 world_params=None,
                 task_params=None,
                 visualize_evaluation=False,
                 initial_seed=0):
        """

        :param evaluation_protocols:
        :param tracker_path:
        :param world_params:
        :param task_params:
        :param visualize_evaluation:
        :param initial_seed:
        """
        self.initial_seed = initial_seed
        self.data_recorder = DataRecorder(output_directory=None)
        if tracker_path is not None:
            self.tracker = Tracker(
                file_path=os.path.join(tracker_path, 'tracker'))
            task_stats = self.tracker.task_stats_log[0]
            del task_stats.task_params['variables_space']
            del task_stats.task_params['task_name']
            self.task = task_generator(task_generator_id=task_stats.task_name,
                                       **task_stats.task_params,
                                       variables_space='space_a_b')
        else:
            if 'variables_space' in task_params:
                del task_params['task_name']
                del task_params['variables_space']
            self.task = task_generator(**task_params,
                                       variables_space='space_a_b')
        if tracker_path:
            if 'seed' in self.tracker.world_params:
                del self.tracker.world_params['seed']
            if 'wrappers' in self.tracker.world_params:
                del self.tracker.world_params['wrappers']
            self.env = CausalWorld(self.task,
                                   **self.tracker.world_params,
                                   seed=self.initial_seed,
                                   data_recorder=self.data_recorder,
                                   enable_visualization=visualize_evaluation)
        else:
            if world_params is not None:
                if 'seed' in world_params:
                    del world_params['seed']
                self.env = CausalWorld(
                    self.task,
                    **world_params,
                    seed=self.initial_seed,
                    data_recorder=self.data_recorder,
                    enable_visualization=visualize_evaluation)
            else:
                self.env = CausalWorld(
                    self.task,
                    seed=self.initial_seed,
                    data_recorder=self.data_recorder,
                    enable_visualization=visualize_evaluation)
        evaluation_episode_length_in_secs = self.task.get_default_max_episode_length(
        )
        self.time_steps_for_evaluation = \
            int(evaluation_episode_length_in_secs / self.env.dt)

        self.evaluation_env = self.env
        self.evaluation_protocols = evaluation_protocols
        self.metrics_list = []
        self.metrics_list.append(MeanFullIntegratedFractionalSuccess())
        self.metrics_list.append(MeanLastIntegratedFractionalSuccess())
        self.metrics_list.append(MeanLastFractionalSuccess())
        return

    def run_episode(self, policy_fn):
        """

        :param policy_fn:

        :return:
        """
        obs = self.evaluation_env.reset()
        for _ in range(self.time_steps_for_evaluation):
            desired_action = policy_fn(obs)
            obs, rew, done, info = self.evaluation_env.step(desired_action)
        return self.data_recorder.get_current_episode()

    def process_metrics(self, episode):
        """

        :param episode:

        :return:
        """
        for metric in self.metrics_list:
            metric.process_episode(episode)
        return

    def get_metric_scores(self):
        """

        :return:
        """
        metrics = dict()
        for metric in self.metrics_list:
            metrics[metric.name] = metric.get_metric_score()
        return metrics

    def reset_metric_scores(self):
        """

        :return:
        """
        for metric in self.metrics_list:
            metric.reset()

    def evaluate_policy(self, policy, fraction=1):
        """

        :param policy:
        :param fraction:

        :return:
        """
        pipeline_scores = dict()
        for evaluation_protocol in self.evaluation_protocols:
            logging.info('Applying the following protocol now, ' + str(evaluation_protocol.get_name()))
            self.evaluation_env = ProtocolWrapper(self.env, evaluation_protocol)
            evaluation_protocol.init_protocol(env=self.env,
                                              tracker=self.env.get_tracker(),
                                              fraction=fraction)
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
            scores['total_resets'] = \
                self.env.get_tracker().get_total_resets()
            pipeline_scores[evaluation_protocol.get_name()] = scores
            self.reset_metric_scores()
        self.evaluation_env.close()
        self.pipeline_scores = pipeline_scores
        return pipeline_scores

    def save_scores(self, evaluation_path, prefix=None):
        """

        :param evaluation_path:
        :param prefix:

        :return:
        """
        if not os.path.isdir(evaluation_path):
            os.makedirs(evaluation_path)
        if prefix is None:
            file_path = os.path.join(evaluation_path, 'scores.json')
        else:
            file_path = os.path.join(evaluation_path,
                                     '{}_scores.json'.format(prefix))
        with open(file_path, "w") as json_file:
            json.dump(self.pipeline_scores, json_file, indent=4)
