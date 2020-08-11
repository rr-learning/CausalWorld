import argparse
import logging
import multiprocessing
from glob import glob

import pandas as pd
import simplejson as json


def flatten_dict(dct):

    # Loop through dict, flatten all subdicts, and then return it
    keys_to_delete = []
    new_dct = {}
    for k in dct:
        v = dct[k]
        if isinstance(v, dict):
            flattened_subdict = flatten_dict(v)
            for sub_k in flattened_subdict:
                new_k = k + '.' + sub_k
                new_dct[new_k] = flattened_subdict[sub_k]
            keys_to_delete.append(k)
    dct.update(new_dct)
    for k in keys_to_delete:
        del dct[k]
    return dct


def _load(path):
    with open(path, 'r') as f:
        result = json.load(f)

    # Flatten results
    result = flatten_dict(result)

    result["path"] = path
    return result


def _get(pattern):
    files = glob(pattern)
    pool = multiprocessing.Pool()
    all_results = pool.map(_load, files)
    return pd.DataFrame(all_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-pattern',
                        default="output/*/evaluation/scores.json",
                        help="glob pattern to all the result files")
    parser.add_argument('--output-path',
                        default="aggregated_results.json",
                        help="path to output json file")
    args = parser.parse_args()

    logging.info("Loading the results.")
    model_results = _get(args.input_file_pattern)
    logging.info("Saving the aggregated results.")
    with open(args.output_path, 'w') as f:
        model_results.to_json(path_or_buf=f)
