from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.qpg.td3 import TD3
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.qpg.td3_agent import Td3Agent
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils import seed as seed_rlpyt
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer

from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.task_generators.task import task_generator
import causal_rl_bench.viewers.task_viewer as viewer
import argparse
import os
import json
import numpy as np
from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.intervention_actors import RandomInterventionActorPolicy, GoalInterventionActorPolicy
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.benchmark.benchmarks import TOWER_2_BENCHMARK
import psutil


world_seed = 0
num_of_envs = 20

NUM_RANDOM_SEEDS = 5
NET_LAYERS = [256, 256]


def save_model_settings(file_path, model_settings):
    with open(file_path, 'w') as fout:
        json.dump(model_settings, fout, indent=4, default=lambda x: x.__dict__)


def baseline_model(model_num):
    benchmarks = sweep('benchmarks', [TOWER_2_BENCHMARK])
    task_configs = [{'task_configs': {'use_train_space_only': True,
                                      'fractional_reward_weight': 0.0,
                                      'dense_reward_weights': [750,
                                                               50,
                                                               250,
                                                               0.005]}}]

    world_params = [{'world_params': {'skip_frame': 3,
                                      'enable_visualization': False,
                                      'observation_mode': 'structured',
                                      'normalize_observations': True,
                                      'action_mode': 'joint_positions'}}]

    random_seeds = sweep('seed', list(range(NUM_RANDOM_SEEDS)))
    algorithms = sweep('algorithm', ['PPO'])

    curriculum_kwargs_1 = {'intervention_actors': [],
                           'actives': []}
    curriculum_kwargs_2 = {'intervention_actors': [GoalInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs_3 = {'intervention_actors': [RandomInterventionActorPolicy()],
                           'actives': [(0, 1e9, 2, 0)]}
    curriculum_kwargs = [curriculum_kwargs_1]

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


def _make_env(rank, model_settings):
    task = task_generator(model_settings['benchmarks']['task_generator_id'], **model_settings['task_configs'])
    env = CausalWorld(task=task,
                      **model_settings['world_params'],
                      seed=0 + rank)
    env = CurriculumWrapper(env,
                            intervention_actors=model_settings["intervention_actors"],
                            actives=model_settings["actives"])
    env.save_world(output_path)
    env = GymEnvWrapper(env)
    return env


def train_model_num(model_settings, output_path):
    seed_rlpyt.set_seed(model_settings['seed'])
    env = _make_env(0, model_settings)
    env.save_world(output_path)
    observation_shape = env.spaces.observation.shape
    output_size = env.spaces.action.shape[0]
    env.close()
    p = psutil.Process()
    cpus = p.cpu_affinity()  # should return a list or a tuple
    # cpus = [0, 1]
    affinity = dict(
        cuda_idx=None,  # whichever one you have
        master_cpus=cpus,
        workers_cpus=list([x] for x in cpus),  # If just one cpu per worker
        set_affinity=True,  # can set to False if you want to turn off rlpyt assigning the psutil cpu_affinity
    )
    sampler = CpuSampler(
        EnvCls=_make_env,
        env_kwargs=dict(rank=0, model_settings=model_settings),
        max_decorrelation_steps=0,
        batch_T=3000,
        batch_B=len(cpus),  # 20 parallel environments.
    )
    model_kwargs = dict(model_kwargs=dict(hidden_sizes=[256, 256]))
    if model_settings['algorithm'] == 'PPO':
        ppo_config = {"discount": 0.98,
                      "entropy_loss_coeff": 2e-4,
                      "learning_rate": 0.00045,
                      "value_loss_coeff": 0.5,
                      "clip_grad_norm": 0.5,
                      "minibatches": 40,
                      "gae_lambda": 0.91,
                      "ratio_clip": 0.4,
                      "epochs": 4}

        algo = PPO(**ppo_config)
        # model_kwargs['model_kwargs'].update({'observation_shape': observation_shape})
        # model_kwargs['model_kwargs'].update({'action_size': output_size})
        # agent = GaussianPgAgent(ModelCls=MujocoFfModel, **model_kwargs)
        agent = MujocoFfAgent(**model_kwargs)
        name = "ppo"

        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=int(300e6),
            log_interval_steps=int(10e6),
            affinity=affinity,
        )
        config = dict(rank=0, model_settings=model_settings)
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', output_path)
        with logger_context(log_dir, 0, name, config, use_summary_writer=True, snapshot_mode='all'):
            runner.train()
        n_itr = runner.get_n_itr()
        return agent, n_itr


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

    agent, n_itr = train_model_num(model_settings, output_path)
    agent.eval_mode(n_itr)
    agent.reset()


    # define a method for the policy fn of your trained model

    def policy_fn(obs, prev_action=None, prev_reward=None):
        if prev_reward is None:
            prev_reward = np.zeros(1, dtype="float32")
        else:
            prev_reward = np.array([prev_reward], dtype="float32")
        if prev_action is None:
            prev_action = buffer_from_example(np.zeros(9, dtype="float32"), 1)
        else:
            prev_action = prev_action.astype('float32')
        obs = obs.astype('float32')
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((obs, prev_action, prev_reward))
        act_pyt, agent_info = agent.step(obs_pyt, act_pyt, rew_pyt)
        action = numpify_buffer(act_pyt)
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
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    evaluator.save_scores(evaluation_path)
    experiments = dict()
    experiments[str(model_num)] = scores
    vis.generate_visual_analysis(evaluation_path, experiments=experiments)
