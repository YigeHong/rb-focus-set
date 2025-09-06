# Experiments for the paper *Unichain and Aperiodicity are Sufficient for Asymptotic Optimality of Average-Reward Restless Bandits*

This project contains the experiment code for the paper: 
Yige Hong, Qiaomin Xie, Yudong Chen, Weina Wang (2024). 
[Unichain and Aperiodicity are Sufficient for Asymptotic Optimality of Average-Reward Restless Bandits](https://arxiv.org/abs/2402.05689).



## How to use 
Python version and required packages:
- python 3.10
- cvxpy 1.3.1
- numpy 1.24.2
- scipy 1.10.1
- matplotlib 3.7.1


The figures will be generated in the folder `figs`.


## File structures
`discrete_RB.py`: 
- Explain each class and how to use them

`rb_settings.py`
- Explain each setting;

`experiments2.py`
- Explain the usage of experiments.py
- Provide parameters for each of the experiments

`understand_id_and_set_expansion.py`
- The experiments comparing the persistency of pibar* under ID policy and set-expansion policy

`understand_assumptions.py`
- Refactor a bit to allow input parameters
- Using the function to plot things ... 

Also some file folders:
- `setting_data`: saved the specification of some RB instances
  - Need to include the setting data for the figures in the paper
- `fig_data`: simulation data
- `random_example_data`: each file contains a large set of random RB instances and simulation results
- `figs2` `figs3` `formal_figs` `formal_figs_exponential`


`plot_figures.py`
- It is a bit ad-hoc. 
- Run the given code to reproduce the experiments, or write your own versions.
- (todo: Provide code for making each figure?)


todo: at least complete the parts correspond to the RB-general paper
the RB-exponential paper requires more work ...


## How to read

The implementations of the simulator and the policies are in the python script `discrete_RB.py`. Specifically,

the important classes in the file `discrete_RB.py` are:

- RB: the simulator of restless bandits

- SingleArmAnalyzer: solving the linear program (3)-(7) 

to generate a priority order of states or the optimal single-armed policy $\bar{\pi}^*$

- PriorityPolicy: the priority policy based on a given priority order

- RandomTBPolicy: the policy that generates virtual actions that each arm wants to follow and break tie uniformly at random

- FTVAPolicy: FTVA($\bar{\pi}^*$)


The usage of those policies can be found in the functions

`Priority_experiment_one_point`

`RandomTBPolicy_experiment_one_point`

`FTVAPolicy_experiment_one_point`

in the file `experiments.py`.


