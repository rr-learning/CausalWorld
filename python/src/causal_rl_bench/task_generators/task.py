from causal_rl_bench.task_generators.pushing import PushingTaskGenerator
from causal_rl_bench.task_generators.picking import PickingTaskGenerator
from causal_rl_bench.task_generators.reaching import ReachingTaskGenerator
from causal_rl_bench.task_generators.stacked_blocks import \
    StackedBlocksGeneratorTask
from causal_rl_bench.task_generators.creative_stacked_blocks import \
    CreativeStackedBlocksGeneratorTask
from causal_rl_bench.task_generators.towers import TowersGeneratorTask
from causal_rl_bench.task_generators.general import GeneralGeneratorTask
from causal_rl_bench.task_generators.pick_and_place import \
    PickAndPlaceTaskGenerator


def task_generator(task_generator_id="reaching", **kwargs):
    """

    :param task_generator_id:
    :param kwargs:

    :return:
    """
    if task_generator_id == "picking":
        task = PickingTaskGenerator(**kwargs)
    elif task_generator_id == "pushing":
        task = PushingTaskGenerator(**kwargs)
    elif task_generator_id == "reaching":
        task = ReachingTaskGenerator(**kwargs)
    elif task_generator_id == "pick_and_place":
        task = PickAndPlaceTaskGenerator(**kwargs)
    elif task_generator_id == "stacked_blocks":
        task = StackedBlocksGeneratorTask(**kwargs)
    elif task_generator_id == "towers":
        task = TowersGeneratorTask(**kwargs)
    elif task_generator_id == "general":
        task = GeneralGeneratorTask(**kwargs)
    elif task_generator_id == "creative_stacked_blocks":
        task = CreativeStackedBlocksGeneratorTask(**kwargs)
    else:
        raise Exception("No valid task_generator_id")
    return task
