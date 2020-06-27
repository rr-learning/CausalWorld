import tensorflow as tf
import tensorflow_hub as hub
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, SAC
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
import argparse
import os
import json
import gym
from gym import spaces
import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_agents import GoalInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.benchmark.benchmarks import PUSHING_BENCHMARK

world_seed = 0
num_of_envs = 20

NUM_RANDOM_SEEDS = 3
NET_LAYERS = [256, 256]
CAMERA_NUMBER = 0


class SingleFingerActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)

    def action(self, act):
        action_to_be_applied = np.copy(act)
        action_to_be_applied[0:3] = [-1.542, -1.28, -2.81]
        action_to_be_applied[6:9] = [-1.542, -1.28, -2.81]
        return action_to_be_applied


class EncoderObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, _representation_function):
        gym.ObservationWrapper.__init__(self, env)
        self.representation_function = _representation_function
        self.observation_space = spaces.Box(low=np.zeros(shape=48),
                                                high=np.ones(shape=48),
                                                dtype=np.float64)

    def observation(self, observation):
        goal_image = observation[3 + CAMERA_NUMBER, ::, ::, :]
        real_image =observation[CAMERA_NUMBER, ::, ::, :]
        image_latents = self.representation_function(real_image, goal_image)
        return np.concatenate(image_latents, axis=None)


class CameraObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, camera_no=0, observation_as_dict=False):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_as_dict = observation_as_dict
        if self.env.env._robot.get_observation_mode():
            self.observation_space = spaces.Box(low=np.zeros(shape=(256, 128, 3)),
                                                high=np.ones(shape=(256, 128, 3)),
                                                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.zeros(shape=(256, 128, 3)),
                                                high=np.full(shape=(256, 128, 3),
                                                             fill_value=255),
                                                dtype=np.uint8)

    def observation(self, observation):
        goal_image = observation[3 + CAMERA_NUMBER, ::, ::, :]
        real_image = observation[CAMERA_NUMBER, ::, ::, :]
        observation_dict = dict()
        observation_dict['goal_image'] = goal_image
        observation_dict['real_image'] = real_image
        if self.observation_as_dict:
            return observation_dict
        else:
            return np.concatenate((real_image, goal_image))


def save_model_settings(file_path, model_settings):
    with open(file_path, 'w') as fout:
        json.dump(model_settings, fout, indent=4, default=lambda x: x.__dict__)


def baseline_model(model_num):
    benchmarks = sweep('benchmarks', [PUSHING_BENCHMARK])

    task_configs = [{'task_configs': {'intervention_split': True,
                                      'training': True,
                                      'sparse_reward_weight': 1}}]

    world_params = [{'world_params': {'skip_frame': 3,
                                      'enable_visualization': True,
                                      'observation_mode': 'cameras',
                                      'normalize_observations': True,
                                      'action_mode': 'joint_positions'}}]

    experiment_mode = sweep('experiment_mode', ['representations', 'end-to-end'])

    random_seeds = sweep('seed', list(range(NUM_RANDOM_SEEDS)))
    algorithms = sweep('algorithm', ['PPO',
                                     'SAC'])
    curriculum_kwargs_1 = {'intervention_actors': [],
                           'actives': []}
    curriculum_kwargs_2 = {'intervention_actors': [GoalInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs = [curriculum_kwargs_1,
                         curriculum_kwargs_2]

    return outer_product([experiment_mode,
                          benchmarks,
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


def get_single_process_env(model_settings, _representation_function):
    task = task_generator(model_settings['benchmarks']['task_generator_id'], **model_settings['task_configs'])
    env = CausalWorld(task=task,
                      **model_settings['world_params'],
                      seed=world_seed)
    env = CurriculumWrapper(env,
                            intervention_actors=model_settings["intervention_actors"],
                            actives=model_settings["actives"])
    env = EncoderObservationWrapper(env, _representation_function)
    env = SingleFingerActionWrapper(env)
    return env


def get_multi_process_env(model_settings, _representation_function):
    def _make_env(rank):
        def _init():
            task = task_generator(model_settings['benchmarks']['task_generator_id'], **model_settings['task_configs'])
            env = CausalWorld(task=task,
                              **model_settings['world_params'],
                              seed=world_seed + rank)
            env = CurriculumWrapper(env,
                                    intervention_actors=model_settings["intervention_actors"],
                                    actives=model_settings["actives"])
            env = EncoderObservationWrapper(env, _representation_function)
            env = SingleFingerActionWrapper(env)
            return env

        return _init

    return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])


def get_SAC_model(model_settings, model_path, _representation_function):
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
    env = get_single_process_env(model_settings, _representation_function)
    if model_settings['experiment_mode'] == 'representations':
        model = SAC(SACMlpPolicy, env, _init_setup_model=True,
                    policy_kwargs=policy_kwargs, **sac_config,
                    verbose=1, tensorboard_log=model_path)
    else:
        model = SAC(CnnPolicy, env, _init_setup_model=True, **sac_config,
                    verbose=1, tensorboard_log=model_path)
    return model, env


def get_PPO_model(model_settings, model_path, _representation_function):
    ppo_config = {"gamma": 0.99,
                  "n_steps": 600,
                  "ent_coef": 0.01,
                  "learning_rate": 0.00025,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "nminibatches": 4,
                  "noptepochs": 4}
    model_settings['train_configs'] = ppo_config
    save_model_settings(os.path.join(model_path, 'model_settings.json'),
                        model_settings)
    env = get_multi_process_env(model_settings, _representation_function)
    if model_settings['experiment_mode'] == 'representations':
        policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=NET_LAYERS)
        model = PPO2(MlpPolicy, env, _init_setup_model=True, policy_kwargs=policy_kwargs,
                     verbose=1, **ppo_config, tensorboard_log=model_path)
    else:
        model = PPO2(CnnPolicy, env, _init_setup_model=True,
                     verbose=1, **ppo_config, tensorboard_log=model_path)
    return model, env


def train_model_num(model_settings, output_path, _representation_function):
    total_time_steps = int(3000000 / 3e2)
    validate_every_timesteps = int(500000 / 5e2)
    model_path = os.path.join(output_path, 'model')
    os.makedirs(model_path)
    set_global_seeds(model_settings['seed'])
    if model_settings['algorithm'] == 'PPO':
        model, env = get_PPO_model(model_settings, model_path, _representation_function)
        num_of_active_envs = num_of_envs
        total_time_steps = 40000000
        validate_every_timesteps = 2000000
    elif model_settings['algorithm'] == 'SAC':
        model, env = get_SAC_model(model_settings, model_path, _representation_function)
        num_of_active_envs = 1
    else:
        model, env = get_PPO_model(model_settings, model_path, _representation_function)
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
    parser.add_argument("--module_path", required=True,
                        help="module path")

    args = vars(parser.parse_args())
    model_num = int(args['model_num'])
    output_path = str(args['output_path'])
    module_path = str(args['module_path'])

    output_path = os.path.join(output_path, str(model_num))
    os.makedirs(output_path)

    model_settings = baseline_model(model_num)
    with hub.eval_function_for_module(module_path) as f:
        def _representation_function(real_image, goal_image):
            """Computes representation vector for input images."""
            output = f(dict(images=(real_image, goal_image)), signature="representation", as_dict=True)
            return np.array(output["default"])


        model = train_model_num(model_settings, output_path, _representation_function)

        # define a method for the policy fn of your trained model
        def policy_fn(observation):
            goal_image = observation[3 + CAMERA_NUMBER, ::, ::, :]
            real_image = observation[CAMERA_NUMBER, ::, ::, :]
            image_latents = _representation_function(real_image, goal_image)
            obs_new = np.concatenate(image_latents, axis=None)
            action = model.predict(obs_new, deterministic=True)[0]
            action[0:3] = [-1.542, -1.28, -2.81]
            action[6:9] = [-1.542, -1.28, -2.81]
            return action

        animation_path = os.path.join(output_path, 'animation')
        os.makedirs(animation_path)
        # Record a video of the policy is done in one line
        viewer.record_video_of_policy(
            task=task_generator(task_generator_id=model_settings['benchmarks']['task_generator_id'],
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
                                       intervention_split=False,
                                       visualize_evaluation=False,
                                       initial_seed=0)
        scores = evaluator.evaluate_policy(policy_fn)
        evaluator.save_scores(evaluation_path)
        experiments = dict()
        experiments[str(model_num)] = scores
        vis.generate_visual_analysis(evaluation_path, experiments=experiments)
