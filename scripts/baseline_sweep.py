import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import TD3, PPO2, SAC, HER
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
import argparse
import os
import json
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_actors import RandomInterventionActorPolicy, GoalInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.wrappers.env_wrappers import HERGoalEnvWrapper
from causal_rl_bench.benchmark.benchmarks import REACHING_BENCHMARK, \
    PUSHING_BENCHMARK, \
    PICKING_BENCHMARK, \
    PICK_AND_PLACE_BENCHMARK, \
    TOWER_2_BENCHMARK
from stable_baselines.ddpg.noise import NormalActionNoise

world_seed = 0
num_of_envs = 2

NUM_RANDOM_SEEDS = 2
NET_LAYERS = [256, 256]


def save_model_settings(file_path, model_settings):
    with open(file_path, 'w') as fout:
        json.dump(model_settings, fout, indent=4, default=lambda x: x.__dict__)


def baseline_model(model_num):
    benchmarks = sweep('benchmarks', [REACHING_BENCHMARK,
                                      PUSHING_BENCHMARK,
                                      PICKING_BENCHMARK,
                                      PICK_AND_PLACE_BENCHMARK,
                                      TOWER_2_BENCHMARK])
    task_configs = [{'task_configs': {'use_train_space_only': True,
                                      'fractional_reward_weight': 1}}]

    world_params = [{'world_params': {'skip_frame': 3,
                                      'enable_visualization': True,
                                      'observation_mode': 'structured',
                                      'normalize_observations': True,
                                      'action_mode': 'joint_positions'}}]

    random_seeds = sweep('seed', list(range(NUM_RANDOM_SEEDS)))
    algorithms = sweep('algorithm', ['PPO',
                                     'SAC',
                                     'TD3',
                                     'SAC_HER'])
    curriculum_kwargs_1 = {'intervention_actors': [],
                           'actives': []}
    curriculum_kwargs_2 = {'intervention_actors': [GoalInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs_3 = {'intervention_actors': [RandomInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs = [curriculum_kwargs_1,
                         curriculum_kwargs_2,
                         curriculum_kwargs_3]

    return outer_product([benchmarks,
                          world_params,
                          task_configs,
                          algorithms,
                          curriculum_kwargs,
                          random_seeds])[model_num]


def sweep(key, values):
    """Sweeps the hyperparameter across different values."""
    return [{key: value} for value in values]


def outer_product(list_of_settings):
    if len(list_of_settings) == 1:
        return list_of_settings[0]
    result = []
    other_items = outer_product(list_of_settings[1:])
    for first_dict in list_of_settings[0]:
        for second_dict in other_items:
            new_dict = {}
            new_dict.update(first_dict)
            new_dict.update(second_dict)
            result.append(new_dict)
    return result


def get_single_process_env(model_settings):
    task = task_generator(model_settings['benchmarks']['task_generator_id'], **model_settings['task_configs'])
    env = CausalWorld(task=task,
                      **model_settings['world_params'],
                      seed=world_seed)
    env = CurriculumWrapper(env,
                            intervention_actors=model_settings["intervention_actors"],
                            actives=model_settings["actives"])
    return env


def get_multi_process_env(model_settings):
    def _make_env(rank):
        def _init():
            task = task_generator(model_settings['benchmarks']['task_generator_id'], **model_settings['task_configs'])
            env = CausalWorld(task=task,
                              **model_settings['world_params'],
                              seed=world_seed + rank)
            env = CurriculumWrapper(env,
                                    intervention_actors=model_settings["intervention_actors"],
                                    actives=model_settings["actives"])
            return env
        return _init

    return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])


def get_TD3_model(model_settings, model_path):
    model_settings['train_configs'] = {}
    policy_kwargs = dict(layers=NET_LAYERS)
    td3_config = {"gamma": 0.98,
                  "tau": 0.01,
                  "learning_rate": 0.00025,
                  "buffer_size": 1000000,
                  "learning_starts": 1000,
                  "batch_size": 256}
    model_settings['train_configs'] = td3_config
    save_model_settings(os.path.join(model_path, 'model_settings.json'),
                        model_settings)
    env = get_single_process_env(model_settings)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3(TD3MlpPolicy, env, action_noise=action_noise, _init_setup_model=True,
                policy_kwargs=policy_kwargs, **td3_config,
                verbose=1, tensorboard_log=model_path)
    return model, env


def get_SAC_model(model_settings, model_path):
    policy_kwargs = dict(layers=NET_LAYERS)
    sac_config = {"gamma": 0.98,
                  "tau": 0.01,
                  "ent_coef": 'auto',
                  "target_entropy": -9,
                  "learning_rate": 0.00025,
                  "buffer_size": 1000000,
                  "learning_starts": 1000,
                  "batch_size": 256}
    model_settings['train_configs'] = sac_config
    save_model_settings(os.path.join(model_path, 'model_settings.json'),
                        model_settings)
    env = get_single_process_env(model_settings)
    model = SAC(SACMlpPolicy, env, _init_setup_model=True,
                policy_kwargs=policy_kwargs, **sac_config,
                verbose=1, tensorboard_log=model_path)
    return model, env


def get_SAC_HER_model(model_settings, model_path):
    model_class = SAC
    policy_kwargs = dict(layers=NET_LAYERS)
    save_model_settings(os.path.join(model_path, 'model_settings.json'),
                        model_settings)
    env = get_single_process_env(model_settings)
    env = HERGoalEnvWrapper(env)
    model = HER('MlpPolicy', env, model_class, _init_setup_model=True,
                policy_kwargs=policy_kwargs,
                n_sampled_goal=4, verbose=1, tensorboard_log=model_path)
    return model, env


def get_PPO_model(model_settings, model_path):
    number_of_time_steps_per_iteration = 120000
    ppo_config = {"gamma": 0.99,
                  "n_steps": int(number_of_time_steps_per_iteration / num_of_envs),
                  "ent_coef": 0.01,
                  "learning_rate": 0.00025,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "nminibatches": 40,
                  "noptepochs": 4}
    model_settings['train_configs'] = ppo_config
    save_model_settings(os.path.join(model_path, 'model_settings.json'),
                        model_settings)
    env = get_multi_process_env(model_settings)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=NET_LAYERS)
    model = PPO2(MlpPolicy, env, _init_setup_model=True, policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config, tensorboard_log=model_path)
    return model, env


def train_model_num(model_settings, output_path):
    total_time_steps = int(3000000)
    validate_every_timesteps = int(500000)
    model_path = os.path.join(output_path, 'model')
    os.makedirs(model_path)
    set_global_seeds(model_settings['seed'])
    if model_settings['algorithm'] == 'PPO':
        model, env = get_PPO_model(model_settings, model_path)
        num_of_active_envs = num_of_envs
        total_time_steps = int(40000000)
        validate_every_timesteps = int(2000000)
    elif model_settings['algorithm'] == 'SAC':
        model, env = get_SAC_model(model_settings, model_path)
        num_of_active_envs = 1
    elif model_settings['algorithm'] == 'TD3':
        model, env = get_TD3_model(model_settings, model_path)
        num_of_active_envs = 1
    elif model_settings['algorithm'] == 'SAC_HER':
        model, env = get_SAC_HER_model(model_settings, model_path)
        num_of_active_envs = 1
    else:
        model, env = get_PPO_model(model_settings, model_path)
        num_of_active_envs = num_of_envs

    checkpoint_callback = CheckpointCallback(save_freq=int(validate_every_timesteps / num_of_active_envs),
                                             save_path=model_path,
                                             name_prefix='model')

    model.learn(int(total_time_steps), callback=checkpoint_callback)
    if env.__class__.__name__ == 'SubprocVecEnv':
        env.env_method("save_world", output_path)
    else:
        env.save_world(output_path)
    env.close()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", required=True, default=0,
                        help="model number")
    parser.add_argument("--output_path", required=True,
                        help="output path")

    args = vars(parser.parse_args())
    model_num = int(args['model_num'])
    output_path = str(args['output_path'])

    output_path = os.path.join(output_path, str(model_num))
    os.makedirs(output_path)

    model_settings = baseline_model(model_num)

    model = train_model_num(model_settings, output_path)


    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]


    animation_path = os.path.join(output_path, 'animation')
    os.makedirs(animation_path)
    # Record a video of the policy is done in one line
    viewer.record_video_of_policy(task=task_generator(task_generator_id=model_settings['benchmarks']['task_generator_id'],
                                                      **model_settings['task_configs']),
                                  world_params=model_settings['world_params'],
                                  policy_fn=policy_fn,
                                  file_name=os.path.join(animation_path, "policy"),
                                  number_of_resets=1,
                                  max_time_steps=600)
    evaluation_path = os.path.join(output_path, 'evaluation')
    os.makedirs(evaluation_path)

    evaluation_protocols = model_settings['benchmarks']['evaluation_protocols']

    evaluator = EvaluationPipeline(evaluation_protocols=
                                   evaluation_protocols,
                                   tracker_path=output_path,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    evaluator.save_scores(evaluation_path)
    experiments = dict()
    experiments[str(model_num)] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
