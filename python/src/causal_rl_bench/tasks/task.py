from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.tasks.example import ExampleTask
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.reaching import ReachingTask
from causal_rl_bench.tasks.pick_and_place import PickAndPlaceTask
from causal_rl_bench.tasks.cuboid_silhouette import CuboidSilhouette


def Task(task_id="picking", **kwargs):
    if task_id == "picking":
        task = PickingTask(**kwargs)
    elif task_id == "pushing":
        task = PushingTask(**kwargs)
    elif task_id == "cuboid_silhouette":
        task = CuboidSilhouette(**kwargs)
    elif task_id == "reaching":
        task = ReachingTask(**kwargs)
    elif task_id == "example":
        task = ExampleTask(**kwargs)
    elif task_id == "pick_and_place":
        task = PickAndPlaceTask(**kwargs)
    else:
        raise Exception("No valid task_id")
    return task
