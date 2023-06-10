rllab is no longer under active development, but an [alliance of researchers](https://github.com/rlworkgroup/) from several universities has adopted it, and now maintains it under the name [**garage**](https://github.com/rlworkgroup/garage).

We recommend you develop new projects, and rebase old ones, onto the actively-maintained [garage](https://github.com/rlworkgroup/garage) codebase, to promote reproducibility and code-sharing in RL research. The new codebase shares almost all of its code with rllab, so most conversions only need to edit package import paths and perhaps update some renamed functions. 

[garage](https://github.com/rlworkgroup/garage) is always looking for new users and contributors, so please consider contributing your rllab-based projects and improvements to the new codebase! Recent improvements include first-class support for TensorFlow, TensorBoard integration, new algorithms including PPO and DDPG, updated Docker images, new environment wrappers, many updated dependencies, and stability improvements.

[![Docs](https://readthedocs.org/projects/rllab/badge)](http://rllab.readthedocs.org/en/latest/)
[![Circle CI](https://circleci.com/gh/rllab/rllab.svg?style=shield)](https://circleci.com/gh/rllab/rllab)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rllab/rllab/blob/master/LICENSE)
[![Join the chat at https://gitter.im/rllab/rllab](https://badges.gitter.im/rllab/rllab.svg)](https://gitter.im/rllab/rllab?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# rllab

rllab is a framework for developing and evaluating reinforcement learning algorithms. It includes a wide range of continuous control tasks plus implementations of the following algorithms:


- [REINFORCE](https://github.com/rllab/rllab/blob/master/rllab/algos/vpg.py)
- [Truncated Natural Policy Gradient](https://github.com/rllab/rllab/blob/master/rllab/algos/tnpg.py)
- [Reward-Weighted Regression](https://github.com/rllab/rllab/blob/master/rllab/algos/erwr.py)
- [Relative Entropy Policy Search](https://github.com/rllab/rllab/blob/master/rllab/algos/reps.py)
- [Trust Region Policy Optimization](https://github.com/rllab/rllab/blob/master/rllab/algos/trpo.py)
- [Cross Entropy Method](https://github.com/rllab/rllab/blob/master/rllab/algos/cem.py)
- [Covariance Matrix Adaption Evolution Strategy](https://github.com/rllab/rllab/blob/master/rllab/algos/cma_es.py)
- [Deep Deterministic Policy Gradient](https://github.com/rllab/rllab/blob/master/rllab/algos/ddpg.py)

rllab is fully compatible with [OpenAI Gym](https://gym.openai.com/). See [here](http://rllab.readthedocs.io/en/latest/user/gym_integration.html) for instructions and examples.

rllab only officially supports Python 3.5+. For an older snapshot of rllab sitting on Python 2, please use the [py2 branch](https://github.com/rllab/rllab/tree/py2).

rllab comes with support for running reinforcement learning experiments on an EC2 cluster, and tools for visualizing the results. See the [documentation](https://rllab.readthedocs.io/en/latest/user/cluster.html) for details.

The main modules use [Theano](http://deeplearning.net/software/theano/) as the underlying framework, and we have support for TensorFlow under [sandbox/rocky/tf](https://github.com/openai/rllab/tree/master/sandbox/rocky/tf).

# Installation
1- Download the zip file or 'git clone https://github.com/rll/rllab'.

2- cd into the rllab-master folder

3- run 'python install setup.py'


# Documentation

Documentation is available online: [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

# Citing rllab

If you use rllab for academic research, you are highly encouraged to cite the following paper:

- Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

# Credits

rllab was originally developed by Rocky Duan (UC Berkeley / OpenAI), Peter Chen (UC Berkeley), Rein Houthooft (UC Berkeley / OpenAI), John Schulman (UC Berkeley / OpenAI), and Pieter Abbeel (UC Berkeley / OpenAI). The library is continued to be jointly developed by people at OpenAI and UC Berkeley.

# Slides

Slides presented at ICML 2016: https://www.dropbox.com/s/rqtpp1jv2jtzxeg/ICML2016_benchmarking_slides.pdf?dl=0
