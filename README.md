# CausalWorld

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/rr-learning/CausalWorld.svg)](https://github.com/rr-learning/CausalWorld/releases)
[![Documentation Status](https://readthedocs.org/projects/causal_world/badge/?version=latest)](https://causal_world.readthedocs.io/en/latest/?badge=latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/rr-learning/CausalWorld/graphs/commit-activity)
[![PR](https://camo.githubusercontent.com/f96261621753dacf526590825b84f87ccb1db0e6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d627269676874677265656e2e7376673f7374796c653d666c6174)](https://github.com/rr-learning/CausalWorld/pulls)
[![Open Source Love png2](https://camo.githubusercontent.com/60dcf2177b53824e7912a6adfb3ff5e318d14ae4/68747470733a2f2f6261646765732e66726170736f66742e636f6d2f6f732f76312f6f70656e2d736f757263652e706e673f763d313033)](https://github.com/rr-learning)


[![<rr-learning>](https://circleci.com/gh/rr-learning/CausalWorld.svg?style=svg&circle-token=28380a46b1bca6dac49b07e423eac9e4111d3d29)](https://circleci.com/gh/rr-learning/CausalWorld)
  
CausalWorld is an open-source framework and benchmark for causal structure learning and testing explicit parameteric generalization in a robotic manipulation environment where tasks range from rather simple to extremely hard, where its crucial to have a deep understanding of the surrounding environment and the meaning of cause and effect. We hope that you find it useful for your own research.  

This package can be either [pip-installed](#install-as-a-pip-package-in-a-conda-env), or used in a catkin workspace.

<p align=center>
<img src="docs/media/pushing.gif" width=200><img src="docs/media/picking.gif" width=200><img src="docs/media/pick_and_place.gif" width=200><img src="docs/media/stacking2.gif" width=200>
</p>

<p align=center>
<img src="docs/media/towers.gif" width=200><img src="docs/media/stacked_blocks.gif" width=200><img src="docs/media/creative_stacked_blocks.gif" width=200><img src="docs/media/general.gif" width=200>
</p>

## Install as a pip package in a conda env

1. Clone this repo and then create it's conda environment to install all dependencies.

  ```bash
  git clone https://github.com/rr-learning/CausalWorld
  cd CausalWorld
  conda env create -f environment.yml OR conda env update --prefix ./env --file environment.yml  --prune
  ```

2. Install the causal_world package inside the (causal_world) conda env.

  ```bash
  conda activate causal_world
  (causal_world) pip install -e .
  ```

3. Make the docs.

  ```bash
  (causal_world) cd docs
  (causal_world) make html
  ```
4. Run the tests.

  ```bash
  (causal_world) python -m unittest discover tests/causal_world/
  ```
  
5. Install other packages for stable baselines (optional)
```bash
(causal_world) pip install tensorflow==1.14.0
(causal_world) pip install stable-baselines==2.10.0
(causal_world) conda install mpi4py
```
  
6. Install other packages for rlkit (optional)

  ```bash
  (causal_world) cd ..
  (causal_world) git clone https://github.com/vitchyr/rlkit.git
  (causal_world) cd rlkit 
  (causal_world) pip install -e .
  (causal_world) pip install torch==1.2.0
  (causal_world) pip install gtimer
  ```

7. Install other packages for viskit (optional)
  ```bash
  (causal_world) cd ..
  (causal_world) git clone https://github.com/vitchyr/viskit.git
  (causal_world) cd viskit 
  (causal_world) pip install -e .
  (causal_world) python viskit/frontend.py path/to/dir/exp*
  ```
  
8. Install other packages for rlpyt (optional)
 ```bash
  (causal_world) cd ..
  (causal_world) git clone https://github.com/astooke/rlpyt.git
  (causal_world) cd rlpyt 
  (causal_world) pip install -e .
  (causal_world) pip install pyprind
  ```



## Getting Started With Couple Of Lines

  ```python
    from causal_world import CausalWorld
    from causal_world import task_generator
    task = task_generator(task_generator_id='general')
    env = CausalWorld(task=task, enable_visualization=True)
    for _ in range(10):
        env.reset()
        for _ in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
    env.close()
  ```
  
## Announcements

## Why would you use CausalWorld for your research?

## Main Features

## Meta-Learning

## Imitation-Learning

## Sim2Real

## Curriculum Through Interventions

## Test How Your Agent Generalizes

## Contributing

## Contact
