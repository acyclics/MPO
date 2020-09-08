# MPO (Maximum a Posteriori Policy Optimization)
Pytorch implementation of MPO (works cited below) with the help of other repositories (also cited below).

Policy evaluation is done using Retrace.

Currently only accommodate Discrete gym environments.

# Usage
Look at main.py for examples of using MPO.

The architectures for Actor and Critic can be changed in mpo_net.py.

# Citations
* Maximum a Posteriori Policy Optimisation (Original MPO algorithm)
>https://arxiv.org/abs/1806.06920

* Relative Entropy Regularized Policy Iteration (Improved MPO algorithm)
>https://arxiv.org/abs/1812.02256

* daisatojp's mpo github repository (MPO implementation as reference)
>https://github.com/daisatojp/mpo

* Openai's ACER github repository (Replay buffer implementation as reference)
>https://github.com/openai/baselines/tree/master/baselines/acer

# Training Results
![mpo_on_LunarLanderV2](lunarlander_mpo.png?raw=true)
