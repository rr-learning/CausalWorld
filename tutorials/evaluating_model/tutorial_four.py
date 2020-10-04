"""
This tutorial shows you how to use a controller and evaluate it afterwards using
an evaluation pipeline compromised of different evaluation protocols.
"""
from causal_world.evaluation.evaluation import EvaluationPipeline
import causal_world.evaluation.protocols as protocols
import causal_world.evaluation.visualization.visualiser as vis

log_relative_path = './reacher_controller_evaluation'


def control_policy(env):
    def _control_policy(obs):
        return \
            env.get_robot().get_joint_positions_from_tip_positions(
                obs[-9:], obs[1:10])
    return _control_policy


def evaluate_controller():
    # pass the different protocols you'd like to evaluate in the following
    task_params = dict()
    task_params['task_generator_id'] = 'reaching'
    world_params = dict()
    world_params['normalize_observations'] = False
    world_params['normalize_actions'] = False
    evaluator = EvaluationPipeline(evaluation_protocols=[
        protocols.ProtocolGenerator(name=
                                    'goal_poses_space_a',
                                    first_level_regex=
                                    'goal_.*',
                                    second_level_regex=
                                    'cylindrical_position',
                                    variable_space='space_a'),
        protocols.ProtocolGenerator(name=
                                    'goal_poses_space_b',
                                    first_level_regex=
                                    'goal_.*',
                                    second_level_regex=
                                    'cylindrical_position',
                                    variable_space='space_b')
    ], task_params=task_params, world_params=world_params,
       visualize_evaluation=True)

    controller_fn = control_policy(evaluator.evaluation_env)
    # For demonstration purposes we evaluate the policy on 10 per
    # cent of the default number of episodes per protocol
    scores = evaluator.evaluate_policy(controller_fn, fraction=0.02)
    evaluator.save_scores(log_relative_path)
    experiments = {'reacher_model': scores}
    vis.generate_visual_analysis(log_relative_path, experiments=experiments)
    print(scores)


if __name__ == '__main__':
    evaluate_controller()
