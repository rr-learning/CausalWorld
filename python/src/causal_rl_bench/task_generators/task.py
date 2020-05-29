from causal_rl_bench.task_generators.pushing import PushingTaskGenerator
from causal_rl_bench.task_generators.example import ExampleTask
from causal_rl_bench.task_generators.picking import PickingTaskGenerator
from causal_rl_bench.task_generators.reaching import ReachingTaskGenerator
from causal_rl_bench.task_generators.stacked_blocks import StackedBlocksTask
from causal_rl_bench.task_generators.stacked_tower import StackedTowerTask
from causal_rl_bench.task_generators.stacked_tower_improper import StackedTowerImproperTask
from causal_rl_bench.task_generators.pyramid import PyramidTask
from causal_rl_bench.task_generators.arch import ArchTask
from causal_rl_bench.task_generators.pick_and_place import PickAndPlaceTask
from causal_rl_bench.task_generators.cuboid_silhouette import CuboidSilhouette


def task_generator(task_generator_id="picking", **kwargs):
    if task_generator_id == "picking":
        task = PickingTaskGenerator(**kwargs)
    elif task_generator_id == "pushing":
        task = PushingTaskGenerator(**kwargs)
    elif task_generator_id == "cuboid_silhouette":
        task = CuboidSilhouette(**kwargs)
    elif task_generator_id == "reaching":
        task = ReachingTaskGenerator(**kwargs)
    elif task_generator_id == "pyramid":
        task = PyramidTask(**kwargs)
    elif task_generator_id == "arch":
        task = ArchTask(**kwargs)
    elif task_generator_id == "example":
        task = ExampleTask(**kwargs)
    elif task_generator_id == "pick_and_place":
        task = PickAndPlaceTask(**kwargs)
    elif task_generator_id == "stacked_blocks":
        task = StackedBlocksTask(**kwargs)
    elif task_generator_id == "stacked_tower":
        task = StackedTowerTask(**kwargs)
    elif task_generator_id == "stacked_tower_improper":
        task = StackedTowerImproperTask(**kwargs)
    else:
        raise Exception("No valid task_generator_id")
    return task
