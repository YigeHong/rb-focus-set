# Experiments for the paper *Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption*

This project contains the experiment code for the paper: 
Yige Hong, Qiaomin Xie, Yudong Chen, Weina Wang (2023). 
[Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption](https://arxiv.org/abs/2306.00196).
*Advances in Neural Information Processing Systems (NeurIPS)* 36, 2023."


## How to use 
Python version and required packages:
- python 3.10
- numpy 1.24.2
- matplotlib 3.7.1
- cvxpy 1.3.1

You can reproduce the Figures in the paper by running the python script `experiments.py`.

The figures will be generated in the folder `figs`.

## How to read
The implementations of the simulator and the policies are in the python script `discrete_RB.py`. Specifically,
the important classes in the file `discrete_RB.py` are:
- RB: the simulator of restless bandits
- SingleArmAnalyzer: solving the linear program (3)-(7) 
to generate a priority order of states or the optimal single-armed policy $\bar{\pi}^*$
- PriorityPolicy: the priority policy based on a given priority order
- RandomTBPolicy: the policy that generates virtual actions that each arm wants to follow and break tie uniformly at random
- FTVAPolicy: our proposed policy FTVA($\bar{\pi}^*$)

The usage of those policies can be found in the functions
`Priority_experiment_one_point`
`RandomTBPolicy_experiment_one_point`
`FTVAPolicy_experiment_one_point`
in the file `experiments.py`.


