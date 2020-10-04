import os
import re
import json
import numpy as np
from stable_baselines.common.callbacks import BaseCallback


def checkpoints_in_folder(folder):

    def is_checkpoint_file(f):
        full_path = os.path.join(folder, f)
        return (os.path.isfile(full_path) and f.startswith("model_") and
                f.endswith('_steps.zip'))

    filenames = [f for f in os.listdir(folder) if is_checkpoint_file(f)]
    regex = re.compile(r'\d+')
    numbers = list([int(regex.search(n).group(0)) for n in filenames])
    assert len(filenames) == len(numbers)
    sorted_idx = np.argsort(numbers)
    numbers = list([numbers[i] for i in sorted_idx])
    filenames = list([filenames[i] for i in sorted_idx])
    return filenames, numbers


def get_latest_checkpoint_path(model_path):
    filenames, numbers = checkpoints_in_folder(model_path)
    if len(filenames) == 0:
        return None, 0
    else:
        ckpt_name = filenames[np.argmax(numbers)]  # latest checkpoint
        ckpt_step = numbers[np.argmax(numbers)]
        ckpt_path = os.path.join(model_path, ckpt_name)
        return ckpt_path, ckpt_step


def save_model_settings(file_path, model_settings):
    # TODO: this needs to be solved in a different way in the future!
    model_settings['intervention_actors'] = [actor.__class__.__name__ for actor
                                             in model_settings['intervention_actors']]
    with open(file_path, 'w') as fout:
        json.dump(model_settings, fout, indent=4, default=lambda x: x.__dict__)


def sweep(key, values):
    return [{key: value} for value in values]


def outer_product(list_of_settings):
    if len(list_of_settings) == 1:
        return list_of_settings[0]
    result = []
    other_items = outer_product(list_of_settings[1:])
    for first_dict in list_of_settings[0]:
        for second_dict in other_items:
            result_dict = dict()
            result_dict.update(first_dict)
            result_dict.update(second_dict)
            result.append(result_dict)
    return result

class PrintTimestepCallback(BaseCallback):

    def _on_step(self) -> bool:
        print(self.model.num_timesteps, flush=True)