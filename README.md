# CausalRLBenchmark {#mainpage}

[TOC]

This package provides the causal rl benchmark. It can be either 
[pip-installed](#install-as-a-pip-package-in-a-conda-env),
or used in a catkin workspace.

![](random_interventions.gif)

![](magic_pick_and_place.gif)

## Install as a pip package in a conda env

1. Clone this repo and then create it's conda environment to install all dependencies.

  ```bash
  git clone https://github.com/rr-learning/CausalRLBench
  cd CausalRLBench
  conda env create -f environment.yml
  ```

2. Install the causal_rl_bench package inside the (causal_rl_bench) conda env.

  ```bash
  conda activate causal_rl_bench
  (causal_rl_bench) pip install -e .
  ```

3. Make the docs.

  ```bash
  (causal_rl_bench) cd docs
  (causal_rl_bench) make html
  ```
4. Run the tests.

  ```bash
  (causal_rl_bench) python -m unittest discover tests/causal_rl_bench/
  ```
  

