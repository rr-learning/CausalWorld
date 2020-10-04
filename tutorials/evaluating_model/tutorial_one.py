"""
This tutorial shows you how to train a policy and evaluate it afterwards using
an evaluation pipeline compromised of different evaluation protocols.
"""

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from causal_world.evaluation.evaluation import EvaluationPipeline
import causal_world.evaluation.protocols as protocols

log_relative_path = './pushing_policy_tutorial_1'


def _make_env(rank):

    def _init():
        task = generate_task(task_generator_id="pushing")
        env = CausalWorld(task=task, enable_visualization=False, seed=rank, skip_frame=3)
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


def evaluate_trained_policy():
    # Load the PPO2 policy trained on the pushing task
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))

    # define a method for the policy fn of your trained model

    def policy_fn(obs):
        return model.predict(obs)[0]

    # pass the different protocols you'd like to evaluate in the following
    evaluator = EvaluationPipeline(evaluation_protocols=[
        protocols.FullyRandomProtocol(name='P11', variable_space='space_b')],
                                   visualize_evaluation=True,
                                   tracker_path=log_relative_path,
                                   initial_seed=0)

    # For demonstration purposes we evaluate the policy on 10 per cent of the default number of episodes per protocol
    scores = evaluator.evaluate_policy(policy_fn, fraction=0.05)
    evaluator.save_scores(log_relative_path)
    print(scores)


if __name__ == '__main__':
    train_policy()
    evaluate_trained_policy()
