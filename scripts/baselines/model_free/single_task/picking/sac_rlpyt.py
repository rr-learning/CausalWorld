from rlpyt.samplers.parallel.cpu.sampler import CpuSampler, CpuResetCollector
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from rlpyt.envs.gym import GymEnvWrapper
import os
import numpy as np
from rlpyt import utils as utils_rlpyt
import psutil


def _make_env(rank):
    task = generate_task(task_generator_id='picking',
                          dense_reward_weights=np.array(
                              [250, 0, 125, 0, 750, 0, 0, 0.005]),
                          fractional_reward_weight=1,
                          goal_height=0.15,
                          tool_block_mass=0.02)
    env = CausalWorld(task=task,
                      skip_frame=3,
                      enable_visualization=False,
                      seed=0,
                      max_episode_length=600)
    env = GymEnvWrapper(env)
    return env


def build_and_train():
    p = psutil.Process()
    cpus = p.cpu_affinity()
    affinity = dict(cuda_idx=None,
                    master_cpus=cpus,
                    workers_cpus=list([x] for x in cpus),
                    set_affinity=True)
    sampler = CpuSampler(EnvCls=_make_env,
                         env_kwargs=dict(rank=0),
                         batch_T=1,
                         batch_B=4,
                         max_decorrelation_steps=0,
                         CollectorCls=CpuResetCollector)
    algo = SAC(batch_size=256,
               min_steps_learn=10000,
               replay_size=1000000,
               replay_ratio=256 / 4,
               target_update_interval=1,
               target_entropy=-9,
               target_update_tau=0.01,
               learning_rate=0.00025,
               action_prior="uniform",
               reward_scale=1,
               reparameterize=True,
               clip_grad_norm=1e9,
               n_step_return=1,
               updates_per_sync=1,
               bootstrap_timelimit=False)  # Run with defaults.
    agent = SacAgent(model_kwargs={'hidden_sizes': [256, 256]})
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=10000,
        affinity=affinity,
    )
    config = dict(env_id='picking')
    name = "sac_rlpyt_picking"
    log_dir = os.path.join(os.path.dirname(__file__), "sac_rlpyt_picking")
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
