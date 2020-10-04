from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import HER, SAC
from stable_baselines.sac.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper

import numpy as np


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, her_config, total_time_steps,
                 validate_every_timesteps, task_name):
    task = generate_task(task_generator_id=task_name,
                          dense_reward_weights=np.array([100000, 0, 0, 0]),
                          fractional_reward_weight=0)
    env = CausalWorld(task=task,
                      skip_frame=skip_frame,
                      enable_visualization=False,
                      seed=seed_num,
                      max_episode_length=maximum_episode_length)
    env = HERGoalEnvWrapper(env)
    env = CurriculumWrapper(env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])
    set_global_seeds(seed_num)
    checkpoint_callback = CheckpointCallback(save_freq=int(
        validate_every_timesteps / num_of_envs),
                                             save_path=log_relative_path,
                                             name_prefix='model')
    model = HER(MlpPolicy,
                env,
                SAC,
                verbose=1,
                policy_kwargs=dict(layers=[256, 256, 256]),
                **her_config,
                seed=seed_num)
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="her_sac",
                callback=checkpoint_callback)
    return


if __name__ == '__main__':
    total_time_steps_per_update = 1000000
    total_time_steps = 60000000
    num_of_envs = 20
    log_relative_path = 'baseline_reaching_her_sac'
    maximum_episode_length = 600
    skip_frame = 3
    seed_num = 0
    task_name = 'reaching'
    her_config = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": 'future',
        "gamma": 0.98,
        "tau": 0.01,
        "ent_coef": 'auto',
        "target_entropy": -9,
        "learning_rate": 0.00025,
        "buffer_size": 1000000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 her_config=her_config,
                 total_time_steps=total_time_steps,
                 validate_every_timesteps=total_time_steps_per_update,
                 task_name=task_name)
