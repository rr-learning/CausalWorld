from causal_rl_bench.evaluation_pipelines.evaluation import EvaluationPipeline
from causal_rl_bench.curriculum.interventions_curriculum import \
    InterventionsCurriculum
from causal_rl_bench.intervention_agents.random import \
    RandomInterventionActorPolicy
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.world import World
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv

log_relative_path = './dummy_policy'


def _make_env(rank):
    def _init():
        task = task_generator(task_generator_id="reaching")
        env = World(task=task, enable_visualization=False,
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
    model = PPO2(MlpPolicy, env, _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1, **ppo_config)
    model.learn(total_timesteps=1000,
                tb_log_name="ppo2",
                reset_num_timesteps=False)
    model.save(os.path.join(log_relative_path, 'model'))
    env.env_method("save_world", log_relative_path)
    env.close()
    return


def evaluate_model():
    # Load the PPO2 policy trained on the cuboid_silhouette task
    model = PPO2.load(os.path.join(log_relative_path, 'model.zip'))
    # define a method for the policy fn of your trained model

    def policy_fn(obs):
        return model.predict(obs)[0]
    #define intervention actors here
    intervention_actors = [RandomInterventionActorPolicy()]
    episode_holds = [5]
    curriculum = InterventionsCurriculum(intervention_actors=
                                         intervention_actors,
                                         episode_holds=episode_holds)
    pipeline = EvaluationPipeline(policy=policy_fn,
                                  testing_curriculum=curriculum,
                                  tracker_path=log_relative_path,
                                  num_seeds=5, episodes_per_seed=20,
                                  intervention_split=False,
                                  training=False, initial_seed=0,
                                  visualize_evaluation=True)
    scores = pipeline.evaluate_policy()
    print(scores)


if __name__ == '__main__':
    #first train the policy, skip if u already trained the policy
    # train_policy()
    evaluate_model()
