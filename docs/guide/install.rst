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

.. code-block:: shell

   git clone git@gitlab.is.tue.mpg.de:robotics/counterfactual.git
   cd counterfactual
   conda env create -f environment.yml

Install the causal_rl_bench package inside the (causal_rl_bench) conda env.

.. code-block:: shell

   conda activate causal_rl_bench
   (causal_rl_bench) python -m pip install .


Make the docs

.. code-block:: shell

  (causal_rl_bench) cd docs
  (causal_rl_bench) make html

Run the tests.

.. code-block:: shell

  (causal_rl_bench) python -m unittest discover tests/causal_rl_bench/


Install other packages for rlkit (optional)

.. code-block:: shell

  (causal_rl_bench) cd ..
  (causal_rl_bench) git clone https://github.com/vitchyr/rlkit.git
  (causal_rl_bench) cd rlkit
  (causal_rl_bench) pip install -e .
  (causal_rl_bench) pip install torch==1.2.0
  (causal_rl_bench) pip install gtimer
