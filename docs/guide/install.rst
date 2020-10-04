.. _install:

===============
Installation
===============

---------------------------------------
Install the latest release through  pip
---------------------------------------

.. code-block:: shell

   pip install causal_world

------------------------------------------
Install from source in a conda environment
------------------------------------------

Clone this repo and then create it's conda environment to install all dependencies.

.. code-block:: shell

   git clone https://github.com/rr-learning/CausalWorld
   cd CausalWorld
   conda env create -f environment.yml

Install the causal_world package inside the (causal_world) conda env.

.. code-block:: shell

   conda activate causal_world
   (causal_world) pip install -e .

You can install the optional packages as indicated `here <https://github.com/rr-learning/CausalWorld>`_