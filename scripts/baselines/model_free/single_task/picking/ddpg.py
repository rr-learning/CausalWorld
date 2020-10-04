from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import mpi4py
from mpi4py import MPI
from stable_baselines.common.callbacks import CheckpointCallback
import numpy as np


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ddpg_config, total_time_steps,
                 validate_every_timesteps, task_name):
    print("Using MPI for multiprocessing with {} workers".format(
        MPI.COMM_WORLD.Get_size()))
    rank = MPI.COMM_WORLD.Get_rank()
    print("Worker rank: {}".format(rank))
    task = generate_task(task_generator_id=task_name,
                          dense_reward_weights=np.array(
                              [250, 0, 125, 0, 750, 0, 0, 0.005]),
                          fractional_reward_weight=1,
                          goal_height=0.15,
                          tool_block_mass=0.02)
    env = CausalWorld(task=task,
                      skip_frame=skip_frame,
                      enable_visualization=False,
                      seed=0,
                      max_episode_length=maximum_episode_length,
                      normalize_actions=False,
                      normalize_observations=False)
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                sigma=float(0.5) *
                                                np.ones(n_actions))
    policy_kwargs = dict(layers=[256, 256])
    checkpoint_callback = CheckpointCallback(save_freq=int(
        validate_every_timesteps / num_of_envs),
                                             save_path=log_relative_path,
                                             name_prefix='model')
    model = DDPG(MlpPolicy,
                 env,
                 verbose=2,
                 param_noise=param_noise,
                 action_noise=action_noise,
                 policy_kwargs=policy_kwargs,
                 **ddpg_config)
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="ddpg",
                callback=checkpoint_callback)
    return


total_time_steps_per_update = 1000000
total_time_steps = 60000000
number_of_time_steps_per_iteration = 120000
num_of_envs = 20
log_relative_path = 'baseline_picking_ddpg_7'
maximum_episode_length = 600
skip_frame = 3
seed_num = 0
task_name = 'picking'
ddpg_config = {
    "gamma": 0.98,
    "tau": 0.01,
    "buffer_size": 1000000,
    # "nb_rollout_steps": 6000,
    # "nb_eval_steps": 600,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "batch_size": 256,
    "tensorboard_log": log_relative_path
}
train_policy(num_of_envs=num_of_envs,
             log_relative_path=log_relative_path,
             maximum_episode_length=maximum_episode_length,
             skip_frame=skip_frame,
             seed_num=seed_num,
             ddpg_config=ddpg_config,
             total_time_steps=total_time_steps,
             validate_every_timesteps=total_time_steps_per_update,
             task_name=task_name)
