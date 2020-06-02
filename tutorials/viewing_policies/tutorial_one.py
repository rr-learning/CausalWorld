from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.agents.reacher_policy import ReacherActorPolicy
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import causal_rl_bench.viewers.task_viewer as viewer


def example():
    # This tutorial shows how to view a pretrained reacher policy
    task = task_generator(task_generator_id='reaching')
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 0
    agent = ReacherActorPolicy()

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return agent.act(obs)

    # Similarly for interactive visualization in the GUI
    viewer.view_policy(task=task,
                       world_params=world_params,
                       policy_fn=policy_fn,
                       max_time_steps=40*960,
                       number_of_resets=40)


if __name__ == '__main__':
    example()
