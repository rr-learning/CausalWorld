.. _getting_started:

===============
Getting Started
===============

Causal World is a new framework that is designed to foster the research on causal structure and transfer
learning of RL agents within a robotics manipulation environment.
Tasks range from simple to extremely challenging.

The goal of the robot in each of the tasks is to
fill a 3d goal shape in the arena with some tool objects available for it to manipulate. Each environment
comprises a simulation version of the "TriFinger robot" that has three fingers - each with 3 joints -
to learn dexterous object manipulation skills with rigid bodies available in a stage.

We provide some task
generators which generates a 3d shape from a specific goal shape distribution which can be rather simple like
pushing, picking, pick and place of single objects but can also be much more involved like stacking towers,
restoring a certain scene of objects or building structures that require imagination.

All environments are compatible with the OpenAI gym interface with a lot of features that you
can use in your research, which means you can start right out of the box.

Setting up an environment can be as simple as two lines of codes

.. code-block:: python

    from causal_world.envs import CausalWorld
    from causal_world.task_generators.task import generate_task

    task = generate_task(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()

By default you are getting a task that has structured observation space and the
low level controller is a joint positions controller. The goal is to solve
the task within (the number of objects in the arena multiplied by 10) seconds.
All of this is easily customizable as you will learn in one of the many tutorials provided.

----------------------------------------------
Basics: Controlling your Environment and Task
----------------------------------------------

.. literalinclude:: ../../tutorials/requesting_task/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/requesting_task/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/requesting_task/tutorial_three.py
   :language: python

-------------------------
Performing Interventions
-------------------------

.. literalinclude:: ../../tutorials/interventions/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_three.py
   :language: python

.. literalinclude:: ../../tutorials/interventions/tutorial_four.py
   :language: python

--------------------------------
Set Up Your Learning Curriculum
--------------------------------

.. literalinclude:: ../../tutorials/curriculums/tutorial_one.py
   :language: python

----------------
Training Agents
----------------

.. literalinclude:: ../../tutorials/stable_baselines/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/stable_baselines/tutorial_two.py
   :language: python

-------------------------------------------
Disentangle the Generalization Capabilities
-------------------------------------------

.. literalinclude:: ../../tutorials/evaluating_model/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/evaluating_model/tutorial_two.py
   :language: python

---------------------------------------------
Viewing and Recording Policies or Logged Data
---------------------------------------------

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_three.py
   :language: python

.. literalinclude:: ../../tutorials/viewing_policies/tutorial_four.py
   :language: python

--------------------------------------
Changing the Action Space of the Robot
--------------------------------------

.. literalinclude:: ../../tutorials/change_action_space/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/change_action_space/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/change_action_space/tutorial_three.py
   :language: python

--------------------------------------------------------
Other Utilities: Model Predictive Control, Logging data
--------------------------------------------------------

.. literalinclude:: ../../tutorials/mpc_w_true_model/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/logging_data/tutorial_one.py
   :language: python

--------------------------------
Much More Available on the Repo
--------------------------------

`here <https://github.com/rr-learning/CausalWorld/tree/master/tutorials>`_
