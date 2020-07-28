.. _getting_started:

===============
Getting Started
===============

Welcome to Causal World!

Causal World is a package that is designed to develop, test and benchmark your RL agents within robotics manipulation
environments - from simple to extremely challenging. Each environments comprises a simulation
version of the "TriFinger robot" that has three fingers - each with 3 joints - to learn dexterous object
manipulation skills with rigid bodies available in a stage. We provide benchmark tasks like reaching, pushing, picking
or pick and place single objects but also much more involved tasks like stacking towers, restoring a certain scene of
objects or building structures that require imagination.

All environments are compatible with the OpenAI gym standard equipped with additional functionality, which means you
can start out of the box with most of your favorite algorithms.

Setting up an environment can be as simple as two lines of codes

.. code-block:: python

    from causal_world import CausalWorld
    from causal_world import task_generator

    task = task_generator(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()

By default you are getting an task that has a feature observation space, takes 9 joint positions for the action
spaces and allows you to solve the task within the number of objects in the arena times 10 seconds. All of this is
easy customizable as you will learn in one of the many tutorials provided.

------------------------------------------------------
Basics: Setting up an environment with different tasks
------------------------------------------------------

.. literalinclude:: ../../tutorials/requesting_task/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/requesting_task/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/requesting_task/tutorial_three.py
   :language: python

-----------------------------------------------------------
Changing the environment instance: Performing interventions
-----------------------------------------------------------

.. literalinclude:: ../../tutorials/interventions/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_three.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_four.py
   :language: python

--------------------------------------
Training agents using stable-baselines
--------------------------------------

.. literalinclude:: ../../tutorials/stable_baselines/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/stable_baselines/tutorial_two.py
   :language: python

---------------------------------------------
Viewing and recording policies or logged data
---------------------------------------------

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_three.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_four.py
   :language: python

-------------------------------------------------
Defining a training curriculum of task variations
-------------------------------------------------

.. literalinclude:: ../../tutorials/curriculums/tutorial_one.py
   :language: python

--------------------------------------
Changing the action space of the robot
--------------------------------------

.. literalinclude:: ../../tutorials/change_action_space/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/change_action_space/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/change_action_space/tutorial_three.py
   :language: python

-------------------------------------------------------------------------------------
Other Utilities: Model Predictive Control, Logging data, loading wrapped environments
-------------------------------------------------------------------------------------

.. literalinclude:: ../../tutorials/mpc_w_true_model/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/logging_data/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/saving_and_loading/tutorial_one.py
   :language: python