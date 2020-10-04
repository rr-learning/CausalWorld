"""
This tutorial shows you how to train a policy and evaluate it afterwards using
one of the default evaluation benchmarks.
"""

from causal_world.evaluation.evaluation import EvaluationPipeline
from causal_world.benchmark.benchmarks import PUSHING_BENCHMARK
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
import causal_world.evaluation.visualization.visualiser as vis

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv

log_relative_path = './pushing_policy_tutorial_3'


def _make_env(rank):

    def _init():
        task = generate_task(task_generator_id="pushing")
        env = CausalWorld(task=task, enable_visualization=False, seed=rank)
        return env

    set_global_seeds(0)
    return _init


def train_policy():
    ppo_config = {
        "gamma": 0.9988,
        "n_steps": 200,
        "ent_coef": 0,
        "learning_rate": 0.001,
        "vf_coef": 0.99,
        "max_grad_norm": 0.1,
        "lam": 0.95,
        "nminibatches": 5,
        "noptepochs": 100,
        "cliprange": 0.2,
        "tensorboard_log": log_relative_path
    }
    os.makedirs(log_relative_path)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(5)])
    model = PPO2(MlpPolicy,
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    model.learn(total_timesteps=1000,
                tb_log_name="ppo2",
                reset_num_timesteps=False)
    model.save(os.path.join(log_relative_path, 'model'))
    env.env_method("save_world", log_relative_path)
    env.close()
    return


def evaluate_model():
    # Load the PPO2 policy trained on the pushing task
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))

    # define a method for the policy fn of your trained model

    def policy_fn(obs):
        return model.predict(obs)[0]

    # Let's evaluate the policy on some default evaluation protocols for reaching task
    evaluation_protocols = PUSHING_BENCHMARK['evaluation_protocols']

    evaluator = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                   tracker_path=log_relative_path,
                                   initial_seed=0)

    # For demonstration purposes we evaluate the policy on 10 per cent of the default number of episodes per protocol
    scores = evaluator.evaluate_policy(policy_fn, fraction=0.02)
    evaluator.save_scores(log_relative_path)
    experiments = {'pushing_model': scores}
    vis.generate_visual_analysis(log_relative_path, experiments=experiments)


if __name__ == '__main__':
    #first train the policy, skip if u already trained the policy
    train_policy()
    evaluate_model()
