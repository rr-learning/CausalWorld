.. _getting_started:

===============
Getting Started
===============

Welcome to CausalRlBench!

CausalRlBench is a package that is designed to develop, test and benchmark your RL agents within robotics manipulation
environments - from simple to extremely challenging. Each environments comprises a simulation
version of the "TriFinger robot" that has three fingers - each with 3 joints - to learn dexterous object
manipulation skills with rigid bodies available in a stage. We provide benchmark tasks like reaching, pushing, picking
or pick and place single objects but also much more involved tasks like stacking towers, restoring a certain scene of
objects or building structures that require imagination.

All environments are compatible with the OpenAI gym standard equipped with additional functionality, which means you
can start out of the box with most of your favorite algorithms.

Setting up an environment can be as simple as two lines of codes

.. code-block:: python

    from causal_rl_bench.envs.causalworld import CausalWorld
    from causal_rl_bench.task_generators.task import task_generator

    task = task_generator(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()

By default you are getting an task that has a feature observation space, takes 9 joint positions for the action
spaces and allows you to solve the task within the number of objects in the arena times 10 seconds. All of this is
easy customizable as you will learn one of the many tutorials provided.
---------------

Demo
----
