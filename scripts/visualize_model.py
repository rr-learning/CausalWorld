from stable_baselines import TD3, PPO2, SAC, HER
from causal_world.task_generators.task import task_generator
import causal_world.viewers.task_viewer as viewer
import argparse
import os
import json


def load_model_settings(file_path):
    with open(file_path, 'r') as fin:
        model_settings = json.load(fin)
    return model_settings


def load_model_from_settings(model_settings, model_path, time_steps):
    algorithm = model_settings['algorithm']
    model = None
    policy_path = os.path.join(model_path,
                               'model_' + str(time_steps) + '_steps')
    if algorithm == 'PPO':
        model = PPO2.load(policy_path)
    elif algorithm == 'SAC':
        model = SAC.load(policy_path)
    elif algorithm == 'TD3':
        model = TD3.load(policy_path)
    elif algorithm == 'SAC_HER':
        model = HER.load(policy_path)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="model path")
    parser.add_argument("--output_path", required=True, help="output path")
    parser.add_argument("--time_steps", required=True, help="time steps")

    args = vars(parser.parse_args())
    time_steps = int(args['time_steps'])
    model_path = str(args['model_path'])
    output_path = str(args['output_path'])

    model_settings = load_model_settings(
        os.path.join(model_path, 'model_settings.json'))

    model = load_model_from_settings(model_settings, model_path, time_steps)

    # define a method for the policy fn of your trained model
    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Record a video of the policy is done in one line
    viewer.record_video_of_policy(task=task_generator(
        task_generator_id=model_settings['benchmarks']['task_generator_id'],
        **model_settings['task_configs']),
                                  world_params=model_settings['world_params'],
                                  policy_fn=policy_fn,
                                  file_name=os.path.join(
                                      output_path,
                                      "policy_{}".format(time_steps)),
                                  number_of_resets=1,
                                  max_time_steps=900)
