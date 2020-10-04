.. _evaluating_policy:

====================
Evaluation Protocols
====================

A central feature of CausalWorld is the explicit parametric formulation of environments
that allow for a precise evaluation of generalization capabilities with respect to any of the
defining environment variables.

In order to evaluate a certain generalization aspect an evaluation protocol needs to be defined.
Each evaluation protocol defines a set of episodes that might differ from each other with respect to the
generalisation aspect under consideration.

If e.g. the agent is evaluated regarding success under different
masses each episodes is a counterfactual version of the other with the mass of a certain object being the
only difference.

After the given set of environments in a protocol are being evaluated,
various aggregated success metrics are being computed based on the fractional reward profiles of each episode.

Each of the environments variables has two associated sets of accessible non-overlapping spaces: A and B.
When a space is activated for the CausalWorld Object during training or evaluation all the variables
will only be allowed to take values within that space, though they generally wont visit the entire space spanned.


-------
Example
-------

As an introductory example we want to show you how you can quantify different generalization aspects depending
on the curriculum exposed during training using the pushing task.

We trained two different agents: First, on the default task only without any type of randomization applied
between episodes. This means the agent always starts with the same initial tool_block pose and the same goal block
pose. Naturally we expect the agent to overfit to this goal over time.

Second, we train an agent using a goal randomization curriculum where we sample new goal poses within space A every episode. Still, we keep the
initial tool_block poses and everything else fixed during training.

As can be seen from the animation below we
see that the former agent overfits to the single goal pose seen during training whereas the later can also push the
tool block towards other goal poses.

Evaluating both agents using the protocols defined in the pushing benchmark
we can quantify and compare performance regarding different aspects in an explicit way.


.. image:: ../media/policy_0.gif
   :scale: 50 %
   :alt: no goal pose randomization
   :align: left

.. image:: ../media/policy_1.gif
   :scale: 50 %
   :alt: goal pose randomization
   :align: left


.. image:: ../media/radar.png
   :scale: 50 %
   :alt: mean last integrated fractional success, radar plot
   :align: center

-----------------------
Using default protocols
-----------------------

Below we show some demo code how you can systematically evaluate agents on a set
of different protocols and visualize the results in radar plots of different scores

.. literalinclude:: ../../tutorials/evaluating_model/tutorial_one.py
   :language: python

.. literalinclude:: ../../tutorials/evaluating_model/tutorial_two.py
   :language: python

.. literalinclude:: ../../tutorials/evaluating_model/tutorial_three.py
   :language: python

------------------------------
Defining a customized protocol
------------------------------