"""
This tutorial shows you how to use a controller and evaluate it afterwards using
an evaluation pipeline compromised of different evaluation protocols.
"""
from causal_world.evaluation.evaluation import EvaluationPipeline
import causal_world.evaluation.protocols as protocols
from causal_world.actors import PushingActorPolicy
import causal_world.evaluation.visualization.visualiser as vis


def evaluate_controller():
    task_params = dict()
    task_params['task_generator_id'] = 'pushing'
    world_params = dict()
    world_params['skip_frame'] = 3
    evaluator = EvaluationPipeline(evaluation_protocols=[
        protocols.FullyRandomProtocol(name='P10',
                                      variable_space='space_a')],
        task_params=task_params, world_params=world_params,
        visualize_evaluation=True)
    policy = PushingActorPolicy()
    scores = evaluator.evaluate_policy(policy.act, fraction=0.1)
    scores = evaluator.evaluate_policy(policy.act, fraction=0.1)
    vis.generate_visual_analysis(log_relative_path, experiments=experiments)
    print(scores)


if __name__ == '__main__':
    evaluate_controller()
