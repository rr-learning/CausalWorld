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

   git clone https://github.com/rr-learning/CausalWorld
   cd CausalWorld
   conda env create -f environment.yml

Install the causal_rl_bench package inside the (causal_rl_bench) conda env.

.. code-block:: shell

   conda activate causal_world
   (causal_world) pip install -e .


Make the docs

.. code-block:: shell

  (causal_world) cd docs
  (causal_world) make html

Run the tests.

.. code-block:: shell

  (causal_world) python -m unittest discover tests/causal_rl_bench/

Install other packages for stable baselines (optional)

.. code-block:: shell

  (causal_world) pip install tensorflow==1.14.0
  (causal_world) pip install stable-baselines==2.10.0
  (causal_world) conda install mpi4py


Install other packages for rlkit (optional)

.. code-block:: shell

  (causal_world) cd ..
  (causal_world) git clone https://github.com/vitchyr/rlkit.git
  (causal_world) cd rlkit
  (causal_world) pip install -e .
  (causal_world) pip install torch==1.2.0
  (causal_world) pip install gtimer
