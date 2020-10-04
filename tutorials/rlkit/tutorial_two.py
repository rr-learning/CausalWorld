"""
This tutorial shows you how to load a picking policy trained with rlkit.
"""
import torch
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
import numpy as np


def simulate_policy():
    file = './her-sac-fetch-experiment/her-sac-fetch-experiment_2020_07_07_11_11_14_0000--s-0/params.pkl'
    data = torch.load(file)
    policy = data['evaluation/policy']
    policy.reset()

    def policy_func(obs):
        # new_obs = np.hstack((obs['observation'], obs['desired_goal']))
        a, agent_info = policy.get_action(obs)
        return a

    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      skip_frame=1,
                      seed=0,
                      max_episode_length=2500)
    env = CurriculumWrapper(env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])
    # env = HERGoalEnvWrapper(env)

    for _ in range(100):
        total_reward = 0
        o = env.reset()
        for _ in range(2500):
            o, reward, done, info = env.step(policy_func(o))
            total_reward += reward
        print("total reward is :", total_reward)
    env.close()


if __name__ == "__main__":
    simulate_policy()
