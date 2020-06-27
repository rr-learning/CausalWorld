from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator


def control_policy(env, obs):
    return \
        env.get_robot().get_joint_positions_from_tip_positions(
            obs[-9:], obs[:9])


def end_effector_pos():
    task = task_generator(task_generator_id='reaching')
    env = CausalWorld(task=task, enable_visualization=True,
                      action_mode="joint_positions", normalize_actions=False,
                      normalize_observations=False)
    obs = env.reset()
    for _ in range(100):
        goal_dict = env.sample_new_goal()
        env.do_intervention(goal_dict)
        obs, reward, done, info = env.step(obs[-9:])
        for _ in range(250):
            obs, reward, done, info = env.step(control_policy(env, obs))
    env.close()


if __name__ == '__main__':
    end_effector_pos()
