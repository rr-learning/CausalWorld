# CausalRLBenchmark {#mainpage}

[TOC]

This package provides the causal rl benchmark. It can be either 
[pip-installed](#install-as-a-pip-package-in-a-conda-env),
or used in a catkin workspace.

## Install as a pip package in a conda env

1. Clone this repo and then create it's conda environment to install all dependencies.

  ```bash
  git clone git@gitlab.is.tue.mpg.de:robotics/counterfactual.git
  cd counterfactual
  conda env create -f environment.yml
  git clone git@git-amd.tuebingen.mpg.de:robotics/pybullet_fingers.git
  cd pybullet_fingers
  git checkout counterfactual_benchmark
  cd ..
  ```

2. Install the causal_rl_bench package inside the (causal_rl_bench) conda env.

  ```bash
  conda activate causal_rl_bench
  (causal_rl_bench) cd counterfactual
  (causal_rl_bench) python -m pip install .
  (causal_rl_bench) cd ../pybullet_fingers
  (causal_rl_bench) python -m pip install .
  ```

2. Make the docs.

  ```bash
  (causal_rl_bench) cd ../counterfactual/docs
  (causal_rl_bench) make html
  (causal_rl_bench) cd ../pybullet_fingers
  (causal_rl_bench) python -m pip install .
  ```

