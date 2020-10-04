from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from causal_world.task_generators.task import generate_task
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.envs.causalworld import CausalWorld
from rlpyt.envs.gym import GymEnvWrapper
import os
import numpy as np
from rlpyt import utils as utils_rlpyt
import psutil


def _make_env(rank):
    task = generate_task('pushing',
                          dense_reward_weights=np.array([2500, 2500, 0]),
                          variables_space='space_a',
                          fractional_reward_weight=100)
    env = CausalWorld(task=task,
                      skip_frame=3,
                      enable_visualization=False,
                      seed=0 + rank)
    env = CurriculumWrapper(env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=(0, 1e9, 2, 0))
    env = GymEnvWrapper(env)
    return env


def build_and_train():
    p = psutil.Process()
    cpus = p.cpu_affinity()
    affinity = dict(cuda_idx=None,
                    master_cpus=cpus,
                    workers_cpus=list([x] for x in cpus),
                    set_affinity=True)
    sampler = CpuSampler(
        EnvCls=_make_env,
        env_kwargs=dict(rank=0),
        max_decorrelation_steps=0,
        batch_T=6000,
        batch_B=len(cpus),  # 20 parallel environments.
    )

    model_kwargs = dict(model_kwargs=dict(hidden_sizes=[256, 256]))
    ppo_config = {
        "discount": 0.98,
        "entropy_loss_coeff": 0.01,
        "learning_rate": 0.00025,
        "value_loss_coeff": 0.5,
        "clip_grad_norm": 0.5,
        "minibatches": 40,
        "gae_lambda": 0.95,
        "ratio_clip": 0.2,
        "epochs": 4
    }

    algo = PPO(**ppo_config)
    agent = MujocoFfAgent(**model_kwargs)

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(60e6),
        log_interval_steps=int(1e6),
        affinity=affinity,
    )
    config = dict(rank=0, env_id='picking')
    name = "ppo_rlpyt_pushing"
    log_dir = os.path.join(os.path.dirname(__file__), name)
    with logger_context(log_dir,
                        0,
                        name,
                        config,
                        use_summary_writer=True,
                        snapshot_mode='all'):
        runner.train()


if __name__ == "__main__":
    utils_rlpyt.seed.set_seed(0)
    build_and_train()
