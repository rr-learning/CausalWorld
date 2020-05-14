from causal_rl_bench.tasks.task import Task
from causal_rl_bench.utils.policy_wrapper import PolicyWrapper
from stable_baselines import PPO2
import causal_rl_bench.viewers.task_viewer as viewer


def example():
    # This tutorial shows how to view policies of trained agents

    task = Task(task_id='pushing')
    world_params = dict()
    world_params["skip_frame"] = 3
    world_params["seed"] = 200
    stable_baselines_policy_path = "/is/cluster/oahmed/Development/cluster_submissions/fred_push_reward_final/saved_model.zip"
    model = PPO2.load(stable_baselines_policy_path)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs)[0]

    # Record a video of the policy is done in one line
    viewer.record_video_of_policy(task=task,
                                  world_params=world_params,
                                  policy_fn=policy_fn,
                                  file_name="pushing_video",
                                  max_time_steps=1000)

    # Similarly for interactive visualization in the GUI
    # viewer.view_policy(task=task,
    #                    world_params=world_params,
    #                    policy_fn=policy_fn,
    #                    max_time_steps=10000,
    #                    number_of_resets=10)


if __name__ == '__main__':
    example()
