import causal_world.evaluation.visualization.visualiser as vis

import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument("--scores_path", required=True, help="scores path")

    args = vars(parser.parse_args())
    scores_path = str(args['scores_path'])
    output_path = str(args['output_path'])

    # with open(scores_path, 'r') as f:
    #     scores = json.load(f)
    #     experiments = {'PPO': scores,
    #                    'SAC': scores,
    #                    'TD3': scores}
    #     vis.generate_visual_analysis(output_path, experiments=experiments)

    PPO_scores_path = 'baselines_picking/5/evaluation/scores.json'
    SAC_scores_path = 'baselines_picking/20/evaluation/scores.json'
    TD3_scores_path = 'baselines_picking/35/evaluation/scores.json'
    experiments = dict()
    with open(PPO_scores_path, 'r') as f:
        scores = json.load(f)
        experiments['PPO'] = scores

    with open(SAC_scores_path, 'r') as f:
        scores = json.load(f)
        experiments['SAC'] = scores

    with open(TD3_scores_path, 'r') as f:
        scores = json.load(f)
        experiments['TD3'] = scores

    vis.generate_visual_analysis(output_path, experiments=experiments)
