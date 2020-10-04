from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import TD3, PPO2, SAC
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper

import util.utils as utils

import os
import numpy as np
import tensorflow as tf


def get_single_process_env(model_settings, model_path, ckpt_step):
    task = generate_task(model_settings['benchmarks']['task_generator_id'],
                          **model_settings['task_configs'])
    env = CausalWorld(task=task,
                      **model_settings['world_params'],
                      seed=model_settings['world_seed'])
    env = CurriculumWrapper(
        env,
        intervention_actors=model_settings["intervention_actors"],
        actives=model_settings["actives"])
    if ckpt_step is None:
        prefix = 0
    else:
        prefix = ckpt_step
    monitor_file = os.path.join(model_path, str(prefix))
    env = Monitor(env, filename=monitor_file,
                  info_keywords=('fractional_success',))

    return env


def get_multi_process_env(model_settings, model_path, num_of_envs, ckpt_step):
    def _make_env(rank):
        def _init():
            task = generate_task(
                model_settings['benchmarks']['task_generator_id'],
                **model_settings['task_configs'])
            env = CausalWorld(task=task,
                              **model_settings['world_params'],
                              seed=model_settings['world_seed'] + rank)
            env = CurriculumWrapper(
                env,
                intervention_actors=model_settings["intervention_actors"],
                actives=model_settings["actives"])
            if ckpt_step is None:
                prefix = 0
            else:
                prefix = ckpt_step
            monitor_file = os.path.join(model_path, str(rank) + '_' + str(prefix))
            env = Monitor(env, filename=monitor_file,
                          info_keywords=('fractional_success',))

            return env

        return _init

    return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])


def get_TD3_model(model_settings, model_path, ckpt_path, ckpt_step, tb_path):
    policy_kwargs = dict(layers=model_settings['NET_LAYERS'])
    env = get_single_process_env(model_settings, model_path, ckpt_step)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))
    if ckpt_path is not None:
        print("Loading model from checkpoint '{}'".format(ckpt_path))
        model = TD3.load(ckpt_path,
                         env=env,
                         _init_setup_model=True,
                         policy_kwargs=policy_kwargs,
                         **model_settings['train_configs'],
                         action_noise=action_noise,
                         verbose=1,
                         tensorboard_log=tb_path)
        model.num_timesteps = ckpt_step
    else:
        model = TD3(TD3MlpPolicy,
                    env,
                    _init_setup_model=True,
                    policy_kwargs=policy_kwargs,
                    action_noise=action_noise,
                    **model_settings['train_configs'],
                    verbose=1,
                    tensorboard_log=tb_path)

    return model, env


def get_SAC_model(model_settings, model_path, ckpt_path, ckpt_step, tb_path):
    policy_kwargs = dict(layers=model_settings['NET_LAYERS'])
    env = get_single_process_env(model_settings, model_path, ckpt_step)
    if ckpt_path is not None:
        print("Loading model from checkpoint '{}'".format(ckpt_path))
        model = SAC.load(ckpt_path,
                         env=env,
                         _init_setup_model=True,
                         policy_kwargs=policy_kwargs,
                         **model_settings['train_configs'],
                         verbose=1,
                         tensorboard_log=tb_path)
        model.num_timesteps = ckpt_step
    else:
        model = SAC(SACMlpPolicy,
                    env,
                    _init_setup_model=True,
                    policy_kwargs=policy_kwargs,
                    **model_settings['train_configs'],
                    verbose=1,
                    tensorboard_log=tb_path)
    return model, env


def get_PPO_model(model_settings, model_path, ckpt_path, ckpt_step, num_of_envs, tb_path):
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=model_settings['NET_LAYERS'])
    env = get_multi_process_env(model_settings, model_path, num_of_envs, ckpt_step)
    if ckpt_path is not None:
        print("Loading model from checkpoint '{}'".format(ckpt_path))
        model = PPO2.load(ckpt_path,
                          env=env,
                          _init_setup_model=True,
                          policy_kwargs=policy_kwargs,
                          **model_settings['train_configs'],
                          verbose=1,
                          tensorboard_log=tb_path)
        model.num_timesteps = ckpt_step
    else:
        model = PPO2(MlpPolicy,
                     env,
                     _init_setup_model=True,
                     policy_kwargs=policy_kwargs,
                     **model_settings['train_configs'],
                     verbose=1,
                     tensorboard_log=tb_path)

    return model, env


def train_model(model_settings, output_path, tensorboard_logging=False):
    num_of_envs = model_settings['num_of_envs']
    model_path = os.path.join(output_path, 'model')

    if tensorboard_logging:
        tb_path = model_path
    else:
        tb_path = None

    try:
        os.makedirs(model_path)
        ckpt_path = None
        ckpt_step = 0
    except FileExistsError:
        print("Folder '{}' already exists".format(model_path))
        ckpt_path, ckpt_step = utils.get_latest_checkpoint_path(model_path)

    set_global_seeds(model_settings['seed'])
    if model_settings['algorithm'] == 'PPO':
        model, env = get_PPO_model(model_settings, model_path, ckpt_path, ckpt_step, num_of_envs, tb_path)
        num_of_active_envs = num_of_envs
        total_time_steps = model_settings['total_time_steps'] - ckpt_step
        validate_every_timesteps = model_settings['validate_every_timesteps']
    elif model_settings['algorithm'] == 'SAC':
        model, env = get_SAC_model(model_settings, model_path, ckpt_path, ckpt_step, tb_path)
        num_of_active_envs = num_of_envs
        total_time_steps = model_settings['total_time_steps'] - ckpt_step
        validate_every_timesteps = model_settings['validate_every_timesteps']
    elif model_settings['algorithm'] == 'TD3':
        model, env = get_TD3_model(model_settings, model_path, ckpt_path, ckpt_step, tb_path)
        num_of_active_envs = num_of_envs
        total_time_steps = model_settings['total_time_steps'] - ckpt_step
        validate_every_timesteps = model_settings['validate_every_timesteps']
    else:
        raise Exception("{} is not supported for training in the baselines".format(model_settings['algorithm']))

    ckpt_frequency = int(validate_every_timesteps / num_of_active_envs)
    checkpoint_callback = CheckpointCallback(save_freq=ckpt_frequency,
                                             save_path=model_path,
                                             name_prefix='model')

    if ckpt_path is None:
        utils.save_model_settings(
            os.path.join(model_path, 'model_settings.json'),
            model_settings)
    model.learn(int(total_time_steps),
                callback=checkpoint_callback,
                reset_num_timesteps=ckpt_path is None)
    model.save(save_path=os.path.join(model_path, 'model_{}_steps'.format(total_time_steps)))
    if env.__class__.__name__ == 'SubprocVecEnv':
        env.env_method("save_world", output_path)
    else:
        env.save_world(output_path)
    env.close()

    return model
