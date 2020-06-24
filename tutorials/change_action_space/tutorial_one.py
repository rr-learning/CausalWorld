from causal_rl_bench.envs.world import World
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.wrappers.action_wrappers import DeltaAction
import numpy as np


def apply_delta_action():
    task = task_generator(task_generator_id='picking')
    env = World(task=task, enable_visualization=True,
                action_mode="end_effector_positions",
                normalize_actions=False,
                normalize_observations=False, skip_frame=1)
    # env = DeltaAction(env)
    for _ in range(50):
        obs = env.reset()
        desired = np.around(obs[18:18+9], decimals=2)
        for _ in range(1000):
            # desired_action = env.get_robot().get_rest_pose()[1]
            # print("desired: ", desired_action)
            # print("observed: ",obs[:9])
            # print("what I wanted to reach", current_obs + desired_action)
            obs, reward, done, info = env.step(desired)
            # print("what I actually reached", np.around(obs[:9], decimals=2))
            # print("diff is", current_obs + desired_action - np.around(obs[:9], decimals=2))
            #     # desired_action = obs[:9]
        # for _ in range(1000):
        #     desired_action[:3] = [0, 0, 0.0825]
        #     # print("desired: ", desired_action)
        #     # print("observed: ",obs[:9])
        #     # print("what I wanted to reach", current_obs + desired_action)
        #     obs, reward, done, info = env.step(desired_action)
        #     print("observed: ", obs[18:18+3])
    env.close()


if __name__ == '__main__':
    apply_delta_action()

