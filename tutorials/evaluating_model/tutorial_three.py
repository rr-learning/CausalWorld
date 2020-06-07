from causal_rl_bench.evaluation.evaluation import EvaluationPipeline
import causal_rl_bench.evaluation.visualization.visualiser as vis
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from causal_rl_bench.task_generators.pushing import PushingTaskGenerator

log_relative_path = './pushing_policy'
# This tutorial is intended to demonstrate the default
# evaluation pipeline for the pushing environment


def _make_env(rank):
    def _init():
        task = task_generator(task_generator_id="pushing")
        env = World(task=task, enable_visualization=False,
                    seed=rank, max_episode_length=960,
                    training=True, intervention_split=True)
        return env
    set_global_seeds(0)
    return _init


def train_policy():
    ppo_config = {"gamma": 0.99,
                  "n_steps": 1000,
                  "ent_coef": 0.01,
                  "learning_rate": 0.00025,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "nminibatches": 4,
                  "noptepochs": 4,
                  "tensorboard_log": log_relative_path}
    os.makedirs(log_relative_path)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(5)])
    model = PPO2(MlpPolicy, env, _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config)
    model.learn(total_timesteps=10000,
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

    default_evaluation_protocols = PushingTaskGenerator.get_default_evaluation_protocols()

    evaluator = EvaluationPipeline(evaluation_protocols=
                                   default_evaluation_protocols,
                                   tracker_path=log_relative_path,
                                   intervention_split=False,
                                   visualize_evaluation=True,
                                   initial_seed=0)
    scores = evaluator.evaluate_policy(policy_fn)
    experiments = dict()
    experiments['PPO_default'] = scores
    vis.generate_visual_analysis('visuals', experiments=experiments)
    print(scores)



if __name__ == '__main__':
    #first train the policy, skip if u already trained the policy
    #train_policy()
    evaluate_model()
