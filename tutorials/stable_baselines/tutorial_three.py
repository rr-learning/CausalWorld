from causal_world.task_generators.task import task_generator
from stable_baselines import SAC
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import task_generator
import causal_world.viewers.task_viewer as viewer


def simulate_policy():
    task = task_generator(task_generator_id='picking')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      skip_frame=3,
                      seed=0,
                      max_episode_length=600)
    file = './model_600000_steps.zip'
    model = SAC.load(file)

    # define a method for the policy fn of your trained model
    def policy_func(obs):
        return model.predict(obs, deterministic=True)[0]

    for _ in range(100):
        total_reward = 0
        o = env.reset()
        for _ in range(600):
            o, reward, done, info = env.step(policy_func(o))
            total_reward += reward
        print("total reward is :", total_reward)
    env.close()


if __name__ == "__main__":
    simulate_policy()
