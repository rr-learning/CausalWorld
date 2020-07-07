import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.wrappers.env_wrappers import HERGoalEnvWrapper
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from causal_rl_bench.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_rl_bench.intervention_actors.goal_actor import GoalInterventionActorPolicy
import os


def experiment(variant):
    # unwrap the TimeLimitEnv wrapper since we manually termiante after 50 steps
    task = task_generator(task_generator_id='reaching')
    eval_env = CausalWorld(task=task, skip_frame=1,
                           enable_visualization=False,
                           seed=0, max_episode_length=1250)
    eval_env = CurriculumWrapper(eval_env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])
    # eval_env = HERGoalEnvWrapper(eval_env)
    task = task_generator(task_generator_id='reaching')
    expl_env = CausalWorld(task=task, skip_frame=1,
                           enable_visualization=False,
                           seed=0, max_episode_length=1250)
    expl_env = CurriculumWrapper(expl_env,
                                 intervention_actors=[GoalInterventionActorPolicy()],
                                 actives=[(0, 1000000000, 1, 0)])
    # expl_env = HERGoalEnvWrapper(expl_env)
    # observation_key = 'observation'
    # desired_goal_key = 'desired_goal'
    #
    # achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    # replay_buffer = ObsDictRelabelingBuffer(
    #     env=eval_env,
    #     observation_key=observation_key,
    #     desired_goal_key=desired_goal_key,
    #     achieved_goal_key=achieved_goal_key,
    #     **variant['replay_buffer_kwargs']
    # )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    # obs_dim = eval_env.observation_space.spaces['observation'].low.size
    # action_dim = eval_env.action_space.low.size
    # goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim ,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim ,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim ,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim ,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim ,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )

    eval_policy = MakeDeterministic(policy)

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )


    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    # trainer = HERTrainer(trainer)
    # eval_path_collector = GoalConditionedPathCollector(
    #     eval_env,
    #     eval_policy,
    #     observation_key=observation_key,
    #     desired_goal_key=desired_goal_key,
    # )
    # expl_path_collector = GoalConditionedPathCollector(
    #     expl_env,
    #     policy,
    #     observation_key=observation_key,
    #     desired_goal_key=desired_goal_key,
    # )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        # algorithm='HER-SAC',
        algorithm='HER',
        version='normal',
        replay_buffer_size=int(1E6),
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=3000,
            num_eval_steps_per_epoch=1250,
            num_expl_steps_per_train_loop=50000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1250,
        ),
        sac_trainer_kwargs=dict(
            discount=0.98,
            soft_target_tau=0.01,
            target_update_period=1,
            policy_lr=0.00025,
            qf_lr=0.00025,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            target_entropy=-9,
        ),
        # replay_buffer_kwargs=dict(
        #     max_size=int(1E6),
        #     fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
        #     fraction_goals_env_goals=0,
        # ),
        qf_kwargs=dict(
            hidden_sizes=[256, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 128],
        ),
    )
    setup_logger('her-sac-fetch-experiment', base_log_dir=os.path.dirname(__file__), variant=variant)
    experiment(variant)