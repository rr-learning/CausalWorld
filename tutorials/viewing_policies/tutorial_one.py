from causal_rl_bench.tasks.task import Task
from causal_rl_bench.viewers.task_viewer import TaskViewer
from causal_rl_bench.utils.policy_wrapper import PolicyWrapper
from stable_baselines import PPO2


#wrap policy here
class StableBaselinePPOPolicy(PolicyWrapper):
    def __init__(self, path):
        super().__init__()
        self.model = PPO2.load(path)

    def get_identifier(self):
        #TODO: discuss with Fred this function
        return "stable_baseline_policy"

    def get_action_for_observation(self, observation):
        return self.model.predict(observation)[0]


def example():
    task = Task(task_id='pushing')
    world_params = dict()
    world_params["skip_frame"] = 1
    world_params["seed"] = 200
    task_viewer = TaskViewer()
    policy_wrapper = StableBaselinePPOPolicy(path="pushing_model.zip")
    task_viewer.view_policy(task=task,
                            world_params=world_params,
                            policy_wrapper=policy_wrapper,
                            max_time_steps=10000)


if __name__ == '__main__':
    example()
