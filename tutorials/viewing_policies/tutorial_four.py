from causal_rl_bench.task_generators.task import task_generator
from stable_baselines import PPO2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import causal_rl_bench.viewers.task_viewer as viewer
from causal_rl_bench.intervention_agents.training_intervention import \
    reset_training_intervention_agent
from causal_rl_bench.wrappers.intervention_wrappers import \
    ResetInterventionsActorWrapper


def example():
    # This tutorial shows how to view policies of trained agents

    task = task_generator(task_generator_id='picking')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 0
    stable_baselines_policy_path = "./saved_model.zip"
    model = PPO2.load(stable_baselines_policy_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    # # Record a video of the policy is done in one line
    # viewer.record_video_of_policy(task=task,
    #                               world_params=world_params,
    #                               policy_fn=policy_fn,
    #                               file_name="pushing_video",
    #                               number_of_resets=10,
    #                               max_time_steps=10*100,
    #                               env_wrappers=np.array(
    #                                   [ResetInterventionsActorWrapper]),
    #                               env_wrappers_args=
    #                               np.array([{'intervention_actor':
    #                                          reset_training_intervention_agent
    #                                          (task_generator_id='reaching')}]))

    # Similarly for interactive visualization in the GUI
    viewer.view_policy(task=task,
                       world_params=world_params,
                       policy_fn=policy_fn,
                       max_time_steps=40*500,
                       number_of_resets=40)
                       # env_wrappers=np.array([ResetInterventionsActorWrapper]),
                       # env_wrappers_args=
                       # np.array([{'intervention_actor':
                       #           reset_training_intervention_agent
                       #           (task_generator_id='picking')}]))


if __name__ == '__main__':
    example()
