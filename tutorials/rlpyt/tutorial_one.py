"""
This tutorial shows you how to train a reaching policy using rlpyt with SAC.
"""
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from rlpyt.envs.gym import GymEnvWrapper
import os


def _make_env(rank):
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      skip_frame=10,
                      enable_visualization=False,
                      seed=0 + rank,
                      max_episode_length=600)
    env = GymEnvWrapper(env)
    return env


def build_and_train():
    affinity = dict(cuda_idx=None, workers_cpus=list(range(15)))
    sampler = CpuSampler(
        EnvCls=_make_env,
        env_kwargs=dict(rank=0),
        batch_T=6000,
        batch_B=20,
    )
    algo = SAC(bootstrap_timelimit=False)  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=600,
        affinity=affinity,
    )
    config = dict(env_id='reaching')
    name = "sac_reaching"
    log_dir = os.path.join(os.path.dirname(__file__), "example")
    with logger_context(log_dir, 0, name, config, use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    build_and_train()
