from causal_rl_bench.evaluation_pipelines.evaluation import EvaluationPipeline
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.envs.world import World
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
import numpy as np
import os
import json
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv

log_relative_path = './trained_policy'


def _make_env(rank):
    def _init():
        task = Task(task_id="reaching")
        env = World(task=task, skip_frame=1,
                    enable_visualization=False,
                    seed=rank, max_episode_length=960)
        return env

    set_global_seeds(0)
    return _init


def train_policy():
    ppo_config = {"gamma": 0.9988,
                  "n_steps": 200,
                  "ent_coef": 0,
                  "learning_rate": 0.001,
                  "vf_coef": 0.99,
                  "max_grad_norm": 0.1,
                  "lam": 0.95,
                  "nminibatches": 5,
                  "noptepochs": 100,
                  "cliprange": 0.2,
                  "tensorboard_log": log_relative_path}
    os.makedirs(log_relative_path)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(5)])
    model = PPO2(MlpPolicy, env, _init_setup_model=True, policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config)
    model.learn(total_timesteps=1000,
                tb_log_name="ppo2",
                reset_num_timesteps=False)
    model.save(os.path.join(log_relative_path, 'model'))
    save_config_file(ppo_config, env)
    env.close()
    return


def save_config_file(ppo_config, env):
    ppo_path = os.path.join(log_relative_path, 'config_ppo.json')
    trackers = env.env_method("get_tracker")
    for i in range(len(trackers)):
        tracker_path = os.path.join(log_relative_path, 'tracker_' + str(i))
        trackers[i].save(file_path=tracker_path)
    with open(ppo_path, 'w') as fout:
        json.dump(ppo_config, fout)


def evaluate_model():
    # Load the PPO2 policy trained on the cuboid_silhouette task
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))
    tracker_1 = Tracker(file_path=os.path.join(log_relative_path, 'tracker_0'))
    # define a method for the policy fn of your trained model

    def policy_fn(obs):
        return model.predict(obs)[0]

    # Run the evaluation pipeline, here for demonstration just cube color is tested
    pipeline = EvaluationPipeline(tracker=tracker_1, policy=policy_fn)
    scores = pipeline.evaluate_reacher_interventions_curriculum()
    print(scores)


def evaluate_model_2():
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    task_params = dict()
    task_params["task_id"] = "reaching"
    # Load the PPO2 policy trained on the cuboid_silhouette task
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    pipeline = EvaluationPipeline(task_params=task_params, policy=policy_fn)
    scores = pipeline.evaluate_reacher_interventions_curriculum()
    print(scores)


def evaluate_model_3():
    from causal_rl_bench.agents.reacher_policy import ReacherPolicy
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["enable_visualization"] = True
    task_params = dict()
    task_params["task_id"] = "reaching"
    # Load the PPO2 policy trained on the cuboid_silhouette task
    reacher_policy = ReacherPolicy()
    # reacher_policy = MovingAverageActionPolicyWrapper(reacher_policy,
    #                                                   widow_size=250)

    def policy_fn(obs):
        return reacher_policy.act(obs)

    pipeline = EvaluationPipeline(task_params=task_params, world_params=world_params, policy=policy_fn)
    scores = pipeline.evaluate_reacher_interventions_curriculum(num_of_episodes=10)
    print(scores)


if __name__ == '__main__':
    #first train the policy, skip if u already trained the policy
    # train_policy()
    evaluate_model_3()
