from causal_rl_bench.agents.reacher_policy import ReacherActorPolicy
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from causal_rl_bench.intervention_agents import GoalInterventionActorPolicy
from causal_rl_bench.curriculum import InterventionsCurriculum
from causal_rl_bench.evaluation_pipelines.evaluation import EvaluationPipeline


def example():
    #define a curriculum for running and testing it
    #lets get the goal intervention agent and impose it every 3 episodes
    curr_curriculum = InterventionsCurriculum(intervention_actors=[GoalInterventionActorPolicy()],
                                              episodes_hold=[3],
                                              timesteps_hold=[None])
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    task_params = dict()
    task_params["task_generator_id"] = "reaching"
    evaluator = EvaluationPipeline(testing_curriculum=curr_curriculum,
                                   world_params=world_params,
                                   task_params=task_params, intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)

    # get policy/controller
    reacher_policy = ReacherActorPolicy()

    def policy_fn(obs):
        return reacher_policy.act(obs)
    evaluator.evaluate_policy(policy_fn)


if __name__ == '__main__':
    example()