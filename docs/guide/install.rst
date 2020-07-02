.. _install:

===============
Installation
===============

-----------
Prerequisites
-----------

-----------
Install as a pip package in a conda env
-----------

Clone this repo and then create it's conda environment to install all dependencies.

.. code-block:: console
   git clone git@gitlab.is.tue.mpg.de:robotics/counterfactual.git
   cd counterfactual
   conda env create -f environment.yml

Install the causal_rl_bench package inside the (causal_rl_bench) conda env.

.. code-block:: console
   conda activate causal_rl_bench
   (causal_rl_bench) python -m pip install .
