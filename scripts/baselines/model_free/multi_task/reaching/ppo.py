from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
import numpy as np
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors.goal_actor import GoalInterventionActorPolicy


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name):

    def _make_env(rank):

        def _init():
            task = generate_task(task_generator_id=task_name,
                                  dense_reward_weights=np.array(
                                      [100000, 0, 0, 0]),
                                  fractional_reward_weight=0)
            env = CausalWorld(task=task,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              max_episode_length=maximum_episode_length)
            env = CurriculumWrapper(
                env,
                intervention_actors=[GoalInterventionActorPolicy()],
                actives=[(0, 1000000000, 1, 0)])

            return env

        set_global_seeds(seed_num)
        return _init

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    checkpoint_callback = CheckpointCallback(save_freq=int(
        validate_every_timesteps / num_of_envs),
                                             save_path=log_relative_path,
                                             name_prefix='model')
    model = PPO2(MlpPolicy,
                 env,
                 _init_setup_model=True,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 **ppo_config)
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="ppo2",
                callback=checkpoint_callback)
    return


if __name__ == '__main__':
    total_time_steps_per_update = 1000000
    total_time_steps = 60000000
    number_of_time_steps_per_iteration = 120000
    num_of_envs = 20
    log_relative_path = 'baseline_reaching_ppo'
    maximum_episode_length = 1000
    skip_frame = 1
    seed_num = 0
    task_name = 'reaching'
    ppo_config = {
        "gamma": 0.99,
        "n_steps": int(number_of_time_steps_per_iteration / num_of_envs),
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "nminibatches": 40,
        "noptepochs": 4,
        "tensorboard_log": log_relative_path
    }
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=total_time_steps,
                 validate_every_timesteps=total_time_steps_per_update,
                 task_name=task_name)
