from causal_rl_bench.evaluation_metrics.evaluation import EvaluationPipeline


def evaluate_model():
    class Tracker:
        def __init__(self, task_params, world_params):
            self.task_params = task_params
            self.world_params = world_params
            self.additional_data = None


    pipeline = EvaluationPipeline(experiment)
    scores = pipeline.evaluate_interventional_robustness()


if __name__ == '__main__':
    evaluate_model()