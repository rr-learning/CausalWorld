import torch
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.intervention_actors import GoalInterventionActorPolicy
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.buffer import torchify_buffer


def simulate_policy():
    task = task_generator(task_generator_id='picking')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      skip_frame=3,
                      seed=0,
                      max_episode_length=600)
    env = GymEnvWrapper(env)
    file = './itr_1097499.pkl'
    data = torch.load(file)
    agent_state_dict = data['agent_state_dict']
    agent = SacAgent(initial_model_state_dict=agent_state_dict)
    agent.initialize(env_spaces=env.spaces)
    agent.eval_mode(itr=data['itr'])

    def policy_func(obs):
        # new_obs = np.hstack((obs['observation'], obs['desired_goal']))
        agent_info = agent.step(torchify_buffer(obs),
                                prev_action=None,
                                prev_reward=None)
        return agent_info.action.numpy()

    # env = HERGoalEnvWrapper(env)
    for _ in range(100):
        total_reward = 0
        o = env.reset()
        for _ in range(600):
            o, reward, done, info = env.step(policy_func(o))
            total_reward += reward
        print("total reward is :", total_reward)
    env.close()


if __name__ == "__main__":
    simulate_policy()
