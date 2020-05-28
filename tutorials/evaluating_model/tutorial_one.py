from causal_rl_bench.evaluation_pipelines.evaluation import EvaluationPipeline
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.tasks.task import Task
from causal_rl_bench.envs.world import World
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import json

log_relative_path = './cs_task_ppo'


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
    task = Task(task_generator_id='cuboid_silhouette', silhouette_size=np.array([1, 2, 1]))
    env = World(task=task, skip_frame=3,
                enable_visualization=False,
                max_episode_length=100)
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

    tracker_path = os.path.join(log_relative_path, 'tracker')
    ppo_path = os.path.join(log_relative_path, 'config_ppo.json')

    tracker = env.get_tracker()
    tracker.save(file_path=tracker_path)
    with open(ppo_path, 'w') as fout:
        json.dump(ppo_config, fout)


def evaluate_model():
    # Train a dummy PPO2 policy on the cuboid_silhouette task
    train_policy()

    # Load the PPO2 policy trained on the cuboid_silhouette task
    # and its associated tracker
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))
    tracker = Tracker(file_path=os.path.join(log_relative_path, 'tracker'))

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs)[0]

    # Run the evaluation pipeline, here for demonstration just cube color is tested
    pipeline = EvaluationPipeline(tracker=tracker, policy=policy_fn)
    scores = pipeline.evaluate_generalisation()

    # Output the results
    for score_axis_key in scores:
        print("Robustness for {}".format(score_axis_key))
        for y in scores[score_axis_key]:
            print(y, ':', scores[score_axis_key][y])

    # This should give the following output which is expected as we are training on non-image data
    # Robustness for cube_color
    # FF0000: {'mean_success': 0.0, 'mean_reward': -104.55712935623171, 'std_reward': 35.496834923811804}
    # 00FF00: {'mean_success': 0.0, 'mean_reward': -104.55712935623171, 'std_reward': 35.496834923811804}
    # 0000FF: {'mean_success': 0.0, 'mean_reward': -104.55712935623171, 'std_reward': 35.496834923811804}


if __name__ == '__main__':
    evaluate_model()