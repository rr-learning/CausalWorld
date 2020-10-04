"""
This tutorial shows you how to use a controller and evaluate it afterwards using
an evaluation pipeline compromised of different evaluation protocols.
"""
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis
from stable_baselines import PPO2


def compare_controllers():
    task_params = dict()
    task_params['task_generator_id'] = 'pushing'
    world_params = dict()
    world_params['skip_frame'] = 3
    evaluation_protocols = PUSHING_BENCHMARK['evaluation_protocols']
    evaluator_1 = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                     task_params=task_params,
                                     world_params=world_params,
                                     visualize_evaluation=False)
    evaluator_2 = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                     task_params=task_params,
                                     world_params=world_params,
                                     visualize_evaluation=False)
    stable_baselines_policy_path_1 = "./model_pushing_curr0.zip"
    stable_baselines_policy_path_2 = "./model_pushing_curr1.zip"
    model_1 = PPO2.load(stable_baselines_policy_path_1)
    model_2 = PPO2.load(stable_baselines_policy_path_2)

    def policy_fn_1(obs):
        return model_1.predict(obs, deterministic=True)[0]

    def policy_fn_2(obs):
        return model_2.predict(obs, deterministic=True)[0]
    scores_model_1 = evaluator_1.evaluate_policy(policy_fn_1, fraction=0.005)
    scores_model_2 = evaluator_2.evaluate_policy(policy_fn_2, fraction=0.005)
    experiments = dict()
    experiments['PPO(0)'] = scores_model_1
    experiments['PPO(1)'] = scores_model_2
    vis.generate_visual_analysis('./', experiments=experiments)


if __name__ == '__main__':
    compare_controllers()
