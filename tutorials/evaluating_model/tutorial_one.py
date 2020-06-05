from causal_rl_bench.agents.reacher_policy import ReacherActorPolicy
from causal_rl_bench.intervention_agents import GoalInterventionActorPolicy, ReacherInterventionActorPolicy
from causal_rl_bench.curriculum import InterventionsCurriculum
from causal_rl_bench.evaluation_pipelines.evaluation import EvaluationPipeline


def evaluate_1():
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    task_params = dict()
    task_params["task_generator_id"] = "reaching"
    evaluator = EvaluationPipeline(intervention_actors=
                                   [ReacherInterventionActorPolicy()],
                                   episodes_hold=[2],
                                   timesteps_hold=[None],
                                   world_params=world_params,
                                   task_params=task_params,
                                   intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)
    # get policy/controller
    reacher_policy = ReacherActorPolicy()

    def policy_fn(obs):
        return reacher_policy.act(obs)

    scores = evaluator.evaluate_policy(policy_fn, num_seeds=1,
                                       episodes_per_seed=10)
    print(scores)


def evaluate_2():
    #define a curriculum for running and testing it
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    task_params = dict()
    task_params["task_generator_id"] = "reaching"
    evaluator = EvaluationPipeline(intervention_actors=
                                   [GoalInterventionActorPolicy()],
                                   episodes_hold=[2],
                                   timesteps_hold=[None],
                                   world_params=world_params,
                                   task_params=task_params,
                                   intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)

    # get policy/controller
    reacher_policy = ReacherActorPolicy()

    def policy_fn(obs):
        return reacher_policy.act(obs)
    #TODO: reset with initial configuration
    scores = evaluator.evaluate_policy(policy_fn,
                                       num_seeds=1,
                                       episodes_per_seed=10)
    print(scores)


def evaluate_3():
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["normalize_actions"] = False
    world_params["normalize_observations"] = False
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    task_params = dict()
    task_params["task_generator_id"] = "reaching"
    evaluator = EvaluationPipeline(intervention_actors=
                                   [GoalInterventionActorPolicy()],
                                   episodes_hold=[2],
                                   timesteps_hold=[None],
                                   world_params=world_params,
                                   task_params=task_params,
                                   intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)

    # get policy/controller
    # reacher_policy = ReacherActorPolicy()
    from causal_rl_bench.envs.world import World
    from causal_rl_bench.task_generators.task import task_generator
    import numpy as np
    task = task_generator(task_generator_id='reaching')
    env = World(task=task, skip_frame=1,
                enable_visualization=False)

    def policy_fn(obs):
        desired_tip_positions = np.array(obs[-9:])
        current_joint_positions = np.array(obs[:9])
        return env.get_robot().\
            get_joint_positions_from_tip_positions(
            desired_tip_positions, current_joint_positions)
    #TODO: reset with initial configuration
    scores = evaluator.evaluate_policy(policy_fn,
                                       num_seeds=5,
                                       episodes_per_seed=10)
    print(scores)


if __name__ == '__main__':
    # evaluate_1()
    # evaluate_2()
    evaluate_3()
