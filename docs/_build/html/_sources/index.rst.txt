Welcome to Causal RL Bench's documentation!
===========================================


Existing benchmarks in reinforcement learning cover a rich and diverse set of
environments and it has been shown that agents can be trained to
solve very challenging tasks. Nevertheless, it is a common problem in RL
that agents are poor at transferring their learned skills to different but
related environments that share a lot of common structure as agents are usually
evaluated on the training distribution itself and similarities to other
environments are ambiguous. We propose a novel benchmark by releasing
various fully parameterized training environments in a robotics setting
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

    from causal_rl_bench.envs.causalworld import CausalWorld
    from causal_rl_bench.task_generators.task import task_generator

    task = task_generator(task_generator_id='stacked_blocks')
    env = CausalWorld(task=task)
    env.reset()
    for _ in range(2000):
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


.. toctree::
   :maxdepth: 2
   :caption: Guide

   guide/install.rst
   guide/getting_started.rst
   guide/task_setups.rst
   guide/evaluating_policy.rst

.. toctree::
   :maxdepth: 3
   :caption: Contents

   modules/causal_rl_bench.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`