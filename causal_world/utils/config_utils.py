from configparser import ConfigParser
import os
from causal_world.loggers.tracker import Tracker
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.wrappers import ObjectSelectorWrapper, DeltaActionEnvWrapper, MovingAverageActionEnvWrapper, \
    HERGoalEnvWrapper, CurriculumWrapper
from causal_world.utils.intervention_actor_utils import initialize_intervention_actors
import copy


def save_config_file(section_names, config_dicts, file_path):
    """

    :param section_names:
    :param config_dicts:
    :param file_path:
    :return:
    """
    config = ConfigParser()
    for i in range(len(section_names)):
        section_name = section_names[i]
        config.add_section(section_name)
        for key, value in config_dicts[i]:
            config.set(section_name, key, value)
    with open(file_path, 'w') as f:
        config.write(f)
    return


def read_config_file(file_path):
    """

    :param file_path:
    :return:
    """
    section_names = []
    config_dicts = []
    config = ConfigParser()
    config.read(file_path)
    for section in config.sections():
        section_names.append(section)
        config_dicts.append(dict())
        for option in config.options(section):
            config_dicts[-1][option] = float(config.get(section, option))
    return section_names, config_dicts


def load_world(tracker_relative_path, enable_visualization=False):
    """
    Loads a world again at the same state as when it was saved.

    :param tracker_relative_path: (str) path specifying where the tracker
                                        saved.
    :param enable_visualization: (bool) True if enabling visualization is
                                        needed.
    :return: (causal_world.CausalWorld) loaded CausalWorld env instance.
    """
    tracker = Tracker(file_path=os.path.join(tracker_relative_path, 'tracker'))
    task_stats = tracker.task_stats_log[0]
    wrapper_dict = copy.deepcopy(tracker.world_params['wrappers'])
    del tracker.world_params['wrappers']
    if 'task_name' in task_stats.task_params:
        del task_stats.task_params['task_name']
    task = generate_task(task_generator_id=task_stats.task_name,
                          **task_stats.task_params)
    env = CausalWorld(task,
                      **tracker.world_params,
                      enable_visualization=enable_visualization)
    for wrapper in wrapper_dict:
        if wrapper == 'object_selector':
            env = ObjectSelectorWrapper(env, **wrapper_dict[wrapper])
        elif wrapper == 'delta_action':
            env = DeltaActionEnvWrapper(env, **wrapper_dict[wrapper])
        elif wrapper == 'moving_average_action':
            env = MovingAverageActionEnvWrapper(env, **wrapper_dict[wrapper])
        elif wrapper == 'her_environment':
            env = HERGoalEnvWrapper(env, **wrapper_dict[wrapper])
        elif wrapper == 'curriculum_environment':
            #first initialize actors
            intervention_actors = \
                initialize_intervention_actors(wrapper_dict[wrapper]['actor_params'])
            #initialize intervention curriculum
            env = CurriculumWrapper(env,
                                    intervention_actors=intervention_actors,
                                    actives=wrapper_dict[wrapper]['actives'])
        else:
            raise Exception("wrapper is not known to be loaded")
    return env
