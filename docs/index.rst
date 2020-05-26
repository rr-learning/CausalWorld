Welcome to Causal RL Bench's documentation!
===========================================


Existing benchmarks in reinforcement learning cover a rich and diverse set of
environments and it has been shown that agents can be trained to
solve very challenging tasks. Nevertheless, it is a common problem in RL
that agents are poor at transferring their learned skills to different but
related environments that share a lot of common structure as agents are usually
evaluated on the training distribution itself and similarities to other
environments are ambiguous. We propose a novel benchmark by releasing X sets of
infinitely many fully parameterized training environments in a robotics setting
which are equipped with unique
sets of testing environments. These environments facilitate a precise evaluation
protocol to test generalisation and robustness capabilities of the acting agents
due to a reformulation of switching between environments through an intervention
on the generative causal model of the environments that allows to quantify the
amount of common shared structure. The skills to learn range from simple to
extremely challenging although the compositional nature of the environments
should allow to reuse previously learned more primitive skills along a
naturally emerging curriculum of tasks.


.. code-block:: python


   number_of_agents = 5
   single_env, parallel_env = EnvironmentWrapper.make_standard_gym_env("Pendulum-v0", random_seed=0,
                                                                    num_of_agents=number_of_agents)
   my_runner = Runner(env=[single_env, parallel_env],
                   log_path=None,
                   num_of_agents=number_of_agents)
   mpc_controller = my_runner.make_mpc_policy(dynamics_function=PendulumTrueModel(),
                                           state_reward_function=pendulum_state_reward_function,
                                           actions_reward_function=pendulum_actions_reward_function,
                                           planning_horizon=30,
                                           optimizer_name='PI2',
                                           true_model=True)

   current_obs = single_env.reset()
   current_obs = np.tile(np.expand_dims(current_obs, 0),
                      (number_of_agents, 1))
   for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_controller.act(current_obs, t)
    current_obs, reward, _, info = single_env.step(action_to_execute[0])
    current_obs = np.tile(np.expand_dims(current_obs, 0),
                          (number_of_agents, 1))
    single_env.render()



   modules/causal_rl_bench.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`