from causal_rl_bench.tasks.pushing import PushingTask
from causal_rl_bench.tasks.example import ExampleTask
from causal_rl_bench.tasks.picking import PickingTask
from causal_rl_bench.tasks.reaching import ReachingTask
from causal_rl_bench.tasks.stacked_blocks import StackedBlocksTask
from causal_rl_bench.tasks.stacked_tower import StackedTowerTask
from causal_rl_bench.tasks.stacked_tower_improper import StackedTowerImproperTask
from causal_rl_bench.tasks.pyramid import PyramidTask
from causal_rl_bench.tasks.arch import ArchTask
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
    elif task_id == "pyramid":
        task = PyramidTask(**kwargs)
    elif task_id == "arch":
        task = ArchTask(**kwargs)
    elif task_id == "example":
        task = ExampleTask(**kwargs)
    elif task_id == "pick_and_place":
        task = PickAndPlaceTask(**kwargs)
    elif task_id == "stacked_blocks":
        task = StackedBlocksTask(**kwargs)
    elif task_id == "stacked_tower":
        task = StackedTowerTask(**kwargs)
    elif task_id == "stacked_tower_improper":
        task = StackedTowerImproperTask(**kwargs)
    else:
        raise Exception("No valid task_id")
    return task
