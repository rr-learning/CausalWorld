from causal_rl_bench.tasks.task import Task
from causal_rl_bench.envs.world import World
import causal_rl_bench.evaluation_metrics.visual_robustness as visual_robustness
import numpy as np


class EvaluationPipeline:
    def __init__(self, policy, tracker, output_path=None, seed=0, num_seeds=50, runs_per_seed=100):
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

    def evaluate_visual_robustness(self):
        pass

    def evaluate_generalisation(self):
        task_stats = self.tracker.task_stats_log[0]
        task = Task(task_id=task_stats.task_name, **task_stats.task_params)
        env = World(task, **self.tracker.world_params, enable_visualization=False, seed=self.seed)
        visual_variables = task.get_visual_variables()
        scores = dict()
        score = dict()
        for key in visual_variables.keys():
            scores_var_sweep = dict()
            for value in visual_variables[key]:
                rews = []
                dones = 0
                for episode_seed in range(self.num_seeds):
                    env.do_intervention(**{key: value})
                    env.seed(episode_seed)
                    obs = env.reset()
                    accumulated_reward = 0
                    for _ in range(self.tracker.world_params["max_episode_length"]):
                        obs, rew, done, info = env.step(self.policy_fn(obs))
                        accumulated_reward += rew
                        if done:
                            dones += 1
                            break
                    rews.append(accumulated_reward)
                score["mean_success"] = float(dones / self.num_seeds)
                score["mean_reward"] = np.mean(rews)
                score["std_reward"] = np.std(rews)
                scores_var_sweep[value] = score
            scores[key] = scores_var_sweep
        return scores

    def evaluate_runtime_interventions(self):
        pass

    def evaluate_confounding_robustness(self):
        pass

    def run_full_evaluation(self):
        pass