from counterfactual.python.src.causal_rl_bench.tasks import stack_cuboid_3d_silhouette as sc_env
from counterfactual.python.src.causal_rl_bench.baselines.model_free.trainers import ppo

import argparse


def train_simple_stacking_task():
    parser = argparse.ArgumentParser(description='train_simple_stacking_task using a model free RL algorithm')
    parser.add_argument('--horizon', type=int, default=1000, help='the horizon of a episode')
    # Add further args here if expected

    args = parser.parse_args()

    # Get an instance of an environment representing a simple task where the goal is to
    # stack two cubes above each other within a predefined 3D silhouette

    env = sc_env.make_env(horizon=args.horizon,
                          enable_visualisation=True,
                          silhouette_size=[1, 1, 2],
                          unit_length=0.03)

    # Instantiate a PPO trainer and train a policy in this env
    trainer = ppo.TrainerPPO(seed=0, env=env)
    trainer.train_policy()
    env.close()


if __name__ == '__main__':
    train_simple_stacking_task()