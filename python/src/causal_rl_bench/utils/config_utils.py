from configparser import ConfigParser
import os
from causal_rl_bench.loggers.tracker import Tracker
from causal_rl_bench.task_generators.task import task_generator
from causal_rl_bench.envs.causalworld import CausalWorld
from causal_rl_bench.wrappers import ObjectSelectorWrapper, DeltaActionEnvWrapper, MovingAverageActionEnvWrapper, \
    HERGoalEnvWrapper, CurriculumWrapper
from causal_rl_bench.utils.intervention_agent_utils import initialize_intervention_agents
from causal_rl_bench.curriculum.curriculum import Curriculum


def save_config_file(section_names, config_dicts, file_path):
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
    tracker = Tracker(file_path=os.path.join(tracker_relative_path,
                                             'tracker'))
    task_stats = tracker.task_stats_log[0]
    task = task_generator(task_generator_id=task_stats._task_name,
                          **task_stats._task_params)
    env = CausalWorld(task, **tracker.world_params,
                      enable_visualization=enable_visualization)
    for wrapper in tracker.world_params['wrappers']:
        if wrapper == 'object_selector':
            env = ObjectSelectorWrapper(env, **tracker.world_params['wrappers'][wrapper])
        elif wrapper == 'delta_action':
            env = DeltaActionEnvWrapper(env, **tracker.world_params['wrappers'][wrapper])
        elif wrapper == 'moving_average_action':
            env = MovingAverageActionEnvWrapper(env, **tracker.world_params['wrappers'][wrapper])
        elif wrapper == 'her_environment':
            env = HERGoalEnvWrapper(env, **tracker.world_params['wrappers'][wrapper])
        elif wrapper == 'curriculum_environment':
            #first initialize actors
            intervention_actors = \
                initialize_intervention_agents(tracker.world_params['wrappers'][wrapper]['agent_params'])
            #initialize intervention curriculum
            env = CurriculumWrapper(env,
                                    intervention_actors=intervention_actors,
                                    episodes_hold=
                                    tracker.world_params['wrappers'][wrapper][
                                        'episodes_hold'],
                                    timesteps_hold=
                                    tracker.world_params['wrappers'][wrapper][
                                        'timesteps_hold']
                                    )
        else:
            raise Exception("wrapper is not known to be loaded")
    return env
