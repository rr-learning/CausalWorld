from causal_world.task_generators.pushing import PushingTaskGenerator
from causal_world.task_generators.picking import PickingTaskGenerator
from causal_world.task_generators.reaching import ReachingTaskGenerator
from causal_world.task_generators.stacked_blocks import \
    StackedBlocksGeneratorTask
from causal_world.task_generators.creative_stacked_blocks import \
    CreativeStackedBlocksGeneratorTask
from causal_world.task_generators.towers import TowersGeneratorTask
from causal_world.task_generators.stacking2 import Stacking2TaskGenerator
from causal_world.task_generators.general import GeneralGeneratorTask
from causal_world.task_generators.pick_and_place import \
    PickAndPlaceTaskGenerator


def generate_task(task_generator_id="reaching", **kwargs):
    """

    :param task_generator_id: picking, pushing, reaching, pick_and_place,
                              stacking2, stacked_blocks, towers, general or
                              creative_stacked_blocks.
    :param kwargs: args that are specific to the task generator

    :return: the task to be used in the CausalWorld
    """
    if task_generator_id == "picking":
        task = PickingTaskGenerator(**kwargs)
    elif task_generator_id == "pushing":
        task = PushingTaskGenerator(**kwargs)
    elif task_generator_id == "reaching":
        task = ReachingTaskGenerator(**kwargs)
    elif task_generator_id == "pick_and_place":
        task = PickAndPlaceTaskGenerator(**kwargs)
    elif task_generator_id == "stacking2":
        task = Stacking2TaskGenerator(**kwargs)
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
