import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from causal_rl_bench.envs.world import World
from causal_rl_bench.tasks.task import Task

seed = 0


def _make_env(rank):
    def _init():
        task = Task(task_id='pushing')
        env = World(task=task, skip_frame=20,
                    enable_visualization=False,
                    seed=seed + rank)
        env.enforce_max_episode_length(episode_length=150)
        return env
    set_global_seeds(seed)
    return _init


def train_policy(num_of_envs):
    total_time_steps = 60000000
    validate_every_timesteps = 1000000
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    model = PPO2(MlpPolicy, env, gamma=0.9988,
                 n_steps=int(150000 / num_of_envs),
                 ent_coef=0,
                 learning_rate=0.001, vf_coef=0.99,
                 max_grad_norm=0.1, lam=0.95, nminibatches=5,
                 noptepochs=100, cliprange=0.2,
                 _init_setup_model=True, policy_kwargs=policy_kwargs,
                 verbose=1,
                 tensorboard_log='./logs')
    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    tb_log_name="ppo2_simple_reward",
                    reset_num_timesteps=False)
        model.save('pushing_model')
    return


if __name__ == '__main__':
    train_policy(num_of_envs=4)

