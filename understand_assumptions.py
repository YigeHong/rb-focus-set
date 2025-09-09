#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates a population of randomly Restless Bandit (RB) instances
and calculates statistics.

This script is part of the experimental analysis for the paper "Unichain and
Aperiodicity are Sufficient for Asymptotic Optimality of Restless Bandits"
(hereafter referred to as "RB-unichain").
"""

__author__ = "Yige Hong"
__date__ = "2025-09-06"
__version__ = "1.0"


import numpy as np

import rb_settings
from rb_settings import *
from discrete_RB import *
from matplotlib import pyplot as plt
import os
from bisect import bisect
import time


def compute_P_and_Phi_eigvals(setting):
    """
    return 1 if unichain and locally stable; return 0 if locally unstable;
    """
    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
    y = analyzer.solve_lp(verbose=False)[1]
    ind_neu = np.where(np.all([y[:,0] > analyzer.EPS, y[:,1] > analyzer.EPS],axis=0))[0]
    if len(ind_neu) > 1:
        return None

    Ppibs_eigs = np.sort(np.abs(np.linalg.eigvals(analyzer.Ppibs)))
    assert np.allclose(Ppibs_eigs[-1], 1)
    Ppibs_second_eig = Ppibs_eigs[-2]

    Phi = analyzer.compute_Phi(verbose=False)
    Phi_spec_rad = np.max(np.abs(np.linalg.eigvals(Phi)))

    return analyzer, Ppibs_second_eig, Phi_spec_rad


def compute_mf_reward(setting, init_state_fracs, priority, T):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, init_state_fracs)
    policy = PriorityPolicy(setting.sspa_size, priority, N=None, act_frac=setting.suggest_act_frac)

    total_reward = 0
    for t in range(T):
        cur_state_fracs = mfrb.get_state_fracs()
        sa_pair_fracs = policy.get_sa_pair_fracs(cur_state_fracs)
        instant_reward = mfrb.step(sa_pair_fracs)
        total_reward += instant_reward
    mfrb_avg_reward = total_reward / T

    return mfrb_avg_reward




def search_and_store_unstable_examples(input_alpha=None, input_plot_cdf=None, input_distr=None):
    """Generates and analyzes populations of random Restless Bandit (RB) instances.

    This script produces the data and plots for Figures 7, 8, 13, and 14
    (the scatter plots and CDFs) in the RB-unichain paper. Simulation data
    is saved in the `random_example_data/` folder at the end of the run.
    The generated figures are saved in `figs2` folder.
    """

    ## Main Configuration
    # --------------------------------------------------------------------------
    # If True, save all instances and simulation results
    update_database = True
    # The total number of random RB instances to generate and store.
    # One can run this function for multiple times with increasing `num_examples`
    # then the new instances will be added to the checkpoint file each time.
    num_examples = 10000

    ## Scatter Plot Settings (Figures 7 & 13)
    # --------------------------------------------------------------------------
    # If True, generates a scatter plot of the second-largest eigenvalue of
    # P_pi vs. the spectral radius of Phi for all generated instances.
    make_scatter_plot = True
    # Filters the scatter plot to only display instances where the second-largest
    # eigenvalue of P_pi is below this threshold.
    unichain_threshold = 0.95

    ## LP & Whittle Index Simulation Settings
    # --------------------------------------------------------------------------
    # If True, runs simulations for the LP and Whittle index policies.
    do_simulation = True if input_plot_cdf is None else input_plot_cdf
    # The number of arms to use in the simulations.
    N = 500
    # The number of simulation steps, in thousands (e.g., 20 means 20,000 steps).
    simu_thousand_steps = 20
    # The list of policies to simulate, which should be a subset of ["lppriority", "whittle"]
    policy_names = ["lppriority", "whittle"]
    # Simulates only the first `simulate_up_to_ith` non-UGAP examples.
    # One can run this function for multiple times with increasing `simulate_up_to_ith`
    simulate_up_to_ith = 10000

    ## CDF Plot Settings (Figures 8 & 14)
    # --------------------------------------------------------------------------
    # NOTE: These options require the LP and Whittle index simulations to be complete.
    # If True, plots the Cumulative Distribution Function (CDF) of the optimality
    # ratio for all locally unstable instances.
    plot_subopt_cdf = True if input_plot_cdf is None else input_plot_cdf
    # If True, saves locally unstable instances with an optimality ratio <= 0.9
    # to the `setting_data/local_unstable_subopt/` directory.
    save_subopt_examples = True
    # Saves the first `i` instances regardless of their properties. Set to -1 to disable.
    save_first_i_examples_unselected = -1
    # Ad-hoc flag, should be set to False for standard runs.
    save_mix_examples = False

    ## FTVA Policy Simulation Settings (Not in Paper)
    # --------------------------------------------------------------------------
    # If True, runs simulations for the FTVA policy on the random instances.
    do_ftva_simulation = False
    # Simulates FTVA for the first `simulate_ftva_up_to_ith` non-UGAP examples.
    # One can run this function for multiple times with increasing `simulate_ftva_up_to_ith`
    simulate_ftva_up_to_ith = 500

    ## Synchronization Time Analysis (Not in Paper)
    # --------------------------------------------------------------------------
    # NOTE: Requires the FTVA simulations to be complete.
    # If True, plots the CDF of synchronization times across all simulated instances.
    plot_sa_hitting_time = False
    # If True, creates a scatter plot of synchronization time vs. optimality ratio.
    plot_sa_hitting_time_vs_opt = False
    # Ad-hoc flags, should be set to False for standard runs.
    find_almost_unstable_examples = False
    save_almost_unstable_examples = False

    ## Mean-Field Dynamics Simulation (Not in Paper)
    # --------------------------------------------------------------------------
    # If True, simulates mean-field dynamics for locally stable examples.
    do_local_stable_simulation = False
    # The number of locally stable examples to simulate.
    simulate_local_stable_up_to_ith = 10000
    # The number of initializations to use when testing for UGAP.
    num_inits_for_test_UGAP = 10
    # The simulation horizon for mean-field dynamics.
    T_mf_simulation = 5000
    # The optimality ratio threshold for defining UGAP suboptimality.
    UGAP_subopt_threshold = 0.99

    ## Instance Generation Hyperparameters
    # --------------------------------------------------------------------------
    # The size of state space
    sspa_size = 10
    # The probability distribution used to generate the random RB instances.
    # Options: "uniform", "dirichlet", "CL", "dirichlet-uniform-nzR0"
    distr = "dirichlet"
    # override if input distribution is specified
    if input_distr is not None:
        distr = input_distr
    # The laziness parameter. If not None, modify the transition probability
    # such that the state stays unchanged with the probability `laziness`.
    laziness = None
    # The sparcity parameter for the Dirichlet distribution.
    alpha = 0.05 if input_alpha is None else input_alpha
    # The parameter for the "CL" distribution; must be > 3.
    beta = 4

    # --------------------------------------------------------------------------
    # Construct a descriptive string that includes the distribution and its
    # key parameters. This is useful for naming files and organizing results.
    # --------------------------------------------------------------------------
    distr_and_parameter = distr
    if distr == "uniform":
        pass
    elif distr in ["dirichlet", "dirichlet-uniform-nzR0"]:
        distr_and_parameter += f"-{alpha}"
    elif distr == "CL":
        distr_and_parameter += f"-{beta}"
    else:
        raise NotImplementedError
    if laziness is not None:
        distr_and_parameter += f"-lazy-{laziness}"


    ## Data Loading and Initialization
    # --------------------------------------------------------------------------
    # Attempt to load a database of previously generated RB instances. If a file
    # matching the current hyperparameters exists, load it. Otherwise, initialize
    # a new data structure to store the results.
    # --------------------------------------------------------------------------
    os.makedirs("random_example_data", exist_ok=True)
    file_path = "random_example_data/random-{}-{}".format(sspa_size, distr_and_parameter)
    if os.path.exists(file_path):
        # Load the existing data file.
        with open(file_path, "rb") as f:
            all_data = pickle.load(f)
        # Verify that the loaded data's hyperparameters match the current settings.
        assert all_data["sspa_size"] == sspa_size
        assert all_data["distr"] == distr
        if distr in ["dirichlet", "dirichlet-uniform-nzR0"]:
            assert all_data["alpha"] == alpha
        if distr == "CL":
            assert all_data["beta"] == beta
        assert all_data["laziness"] == laziness
        assert "examples" in all_data
    else:
        # If no data file exists, create a new dictionary to hold the data.
        all_data = {}
        all_data["sspa_size"] = sspa_size
        all_data["distr"] = distr
        all_data["alpha"] = alpha
        all_data["beta"] = beta
        all_data["laziness"] = laziness
        all_data["examples"] = []
        all_data["is_sa"] = []

    ## Simulation Data Initialization
    # --------------------------------------------------------------------------
    # The following blocks load saved simulation data if it exists. If not, they
    # initialize the data structures needed to store new simulation results.
    # --------------------------------------------------------------------------

    # --- Load LP Index and Whittle Index Simulation Data ---
    simu_file_path = "random_example_data/simu-N{}-random-{}-{}".format(N, sspa_size, distr_and_parameter)
    if os.path.exists(simu_file_path):
        with open(simu_file_path, "rb") as f:
            simu_data = pickle.load(f)
    else:
        simu_data = {}
        simu_data["N"] = N
        # simu_data[policy_name][i] is the reward trace for instance `i` under policy `policy_name`;
        # it is a list containing average reward every thousant time step
        # simu_data[(policy+"-final-state")] stores the final state of the last simulation,
        # i.e., the state at 1000*len(simu_data["LP-index"][i]) time step
        for policy_name in policy_names:
            simu_data[policy_name] = [[] for i in range(num_examples)]
            simu_data[(policy_name+"-final-state")] =  [None for i in range(num_examples)]

    # --- Load FTVA Policy Simulation Data ---
    simu_ftva_file_path = "random_example_data/simu-ftva-N{}-random-{}-{}".format(N, sspa_size, distr_and_parameter)
    if os.path.exists(simu_ftva_file_path):
        with open(simu_ftva_file_path, "rb") as f:
            simu_ftva_data = pickle.load(f)
    else:
        simu_ftva_data = {}
        simu_ftva_data["N"] = N
        simu_ftva_data["FTVA"] = [[] for i in range(num_examples)]
        simu_ftva_data["FTVA-final-state"] = [None for i in range(num_examples)]

    # --- Load Mean-Field Simulation Data for Locally Stable Examples ---
    simu_local_stable_path = "random_example_data/simu-local-stable-N{}-random-{}-{}".format(N, sspa_size, distr_and_parameter)
    if os.path.exists(simu_local_stable_path):
        with open(simu_local_stable_path, "rb") as f:
            simu_local_stable_data = pickle.load(f)
    else:
        simu_local_stable_data = {}
        simu_local_stable_data["N"] = N
        for policy_name in policy_names:
            simu_local_stable_data[policy_name] = [[] for i in range(num_examples)]


    ## Generate and Analyze New RB Instances
    # --------------------------------------------------------------------------
    # If the number of existing instances is less than `num_examples`, generate
    # new ones until the target number is reached. For each new instance, key
    # properties like second-largest eigenvalue,  LP upper bound, LP solution
    # are pre-calculated and stored.
    # --------------------------------------------------------------------------
    num_exist_examples = len(all_data["examples"])
    if num_exist_examples < num_examples:
        # Pre-calculate the B matrix for the Chung-Lu model if needed.
        if distr == "CL":
            B = ChungLuBeta2B(beta)
        # Loop to create and analyze new instances.
        for i in range(num_examples):
             # Generate a new RB instance based on the chosen distribution.
            if distr in ["uniform", "dirichlet", "dirichlet-uniform-nzR0"]:
                setting = RandomExample(sspa_size, distr, laziness=laziness, parameters=[alpha])
            elif distr == "CL":
                setting = ChungLuRandomExample(sspa_size, beta, B)
                print_bandit(setting)
            else:
                raise NotImplementedError
            # Analyze the instances. Skip instances that have already been generated in previous runs.
            if i > num_exist_examples:
                result = compute_P_and_Phi_eigvals(setting)
                # If multiple neutral states, `compute_P_and_Phi_eigvals` returns None, skip this instance
                if result is None:
                    all_data["examples"].append(None)
                    continue
                # calculate key properties of the instances
                analyzer = result[0]
                setting.avg_reward_upper_bound = analyzer.opt_value
                setting.y = analyzer.y.value.copy()
                setting.lp_priority = analyzer.solve_LP_Priority(verbose=False)
                setting.whittle_priority = analyzer.solve_whittles_policy()
                setting.unichain_eigval = result[1]
                setting.local_stab_eigval = result[2]
                print(f"{i}-th example, LP upper bound={setting.avg_reward_upper_bound}, LP Priority={setting.lp_priority}, "
                      f"Whittle priority={setting.whittle_priority}, P_pi_radius={setting.unichain_eigval}, Phi_radius={setting.local_stab_eigval}")
                all_data["examples"].append(setting)


    ## ---calculate the average sparsity---
    P_nnz_entries_total = 0
    num_rows_total = 0
    for i in range(len(all_data["examples"])):
        setting = all_data["examples"][i]
        if setting is not None:
            P_nnz_entries_total += np.sum(setting.trans_tensor > 0)
            num_rows_total += setting.sspa_size**2 * 2
    print(f"Average percent of non-zero entries = {P_nnz_entries_total / num_rows_total}")


    ## ---For the locally unstable examples, run the mean-field limit reward---
    for i, setting in enumerate(all_data["examples"]):
        if setting is None:
            continue
        if (setting.local_stab_eigval > 1) and (setting.unichain_eigval < 1):
            if setting.avg_reward_lpp_mf_limit is None:
                print("analyzing the {}-th example's LP-Priority".format(i))
                # the subroutine of computing mf limit
                init_state_frac = np.ones((sspa_size,)) / sspa_size
                setting.avg_reward_lpp_mf_limit = compute_mf_reward(setting, init_state_frac, setting.lp_priority, T_mf_simulation)
            if setting.avg_reward_whittle_mf_limit is None:
                print("analyzing the {}-th example's Whittle index".format(i))
                init_state_frac = np.ones((sspa_size,)) / sspa_size
                if type(setting.whittle_priority) == int and (setting.whittle_priority < 0):
                    setting.avg_reward_whittle_mf_limit = 0
                else:
                    setting.avg_reward_whittle_mf_limit = compute_mf_reward(setting, init_state_frac, setting.whittle_priority, T_mf_simulation)


    ## ---calculate hitting time to determine if the example satisfies SA---
    max_recent_hitting_time = 0
    for i, setting in enumerate(all_data["examples"]):
        if i%100 == 0:
            print("computing SA max hitting time, {} examples finished; largest hitting time recently = {}".format(i, max_recent_hitting_time))
            max_recent_hitting_time = 0
        if setting is None:
            continue
        if all_data["examples"][i].sa_max_hitting_time is not None:
            max_recent_hitting_time = max(max_recent_hitting_time, setting.sa_max_hitting_time)
            continue
        d = setting.sspa_size
        analyzer = SingleArmAnalyzer(d, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
        y = analyzer.solve_lp(verbose=False)[1]
        # test the time and correctness of two ways of computing the joint transition matrix
        # joint_trans_mat[m,j,k,l] represents the probability of going from joint state (m,j) to (k,l),
        # where the first entry in the joint state belongs to the leader arm, the second entry belongs to the follower arm
        # joint_trans_mat[m,j,k,l] = Ppibs[m,k] * sum_a P(j,a,l) pibs(m,a);
        joint_trans_mat = np.zeros((d,d,d,d))
        P_temp = np.tensordot(analyzer.Ppibs, analyzer.trans_tensor, axes=0)
        for m in range(d):
            joint_trans_mat[m,:,:,:] = np.tensordot(P_temp[m,:,:,:,:], analyzer.policy[m,:], axes=([2], [0]))
        joint_trans_mat = joint_trans_mat.transpose((0,2,1,3))
        # set transition prob in absorbing states to zero
        for m in range(d):
            joint_trans_mat[m,m,:,:] = 0
        # the cost vector for computing hitting time
        cost_vec = np.ones((d, d)) - np.eye(d)
        h = np.linalg.solve(np.eye(d**2) - joint_trans_mat.reshape(d**2, d**2), cost_vec.reshape((-1,)))
        all_data["examples"][i].sa_max_hitting_time = np.max(h)
        max_recent_hitting_time = max(max_recent_hitting_time, setting.sa_max_hitting_time)
        if all_data["examples"][i].sa_max_hitting_time > 1e4:
            print("The {}-th example is non-SA, max hitting time = {}".format(i, all_data["examples"][i].sa_max_hitting_time))


    ## ---make scatter plot of unichain and local stability---
    if make_scatter_plot:
        eig_vals_list = []
        unstable_unichain_count = 0
        unichain_count = 0
        for i, setting in enumerate(all_data["examples"]):
            if setting is None:
                continue
            if np.any(setting.y[:,0]+setting.y[:,1] < 1e-7):
                # skip the ambiguous examples where the definition of local stability is not unique
                continue
            eig_vals_list.append([setting.unichain_eigval, setting.local_stab_eigval])
            if setting.unichain_eigval <= unichain_threshold:
                unichain_count +=1
                if setting.local_stab_eigval > 1:
                    unstable_unichain_count += 1
        print("Total number of non-ambiguous examples= {}, 0.95-unichain_count = {}, unstable-0.95-unichain_count = {}".format(
            len(eig_vals_list), unichain_count, unstable_unichain_count))
        eig_vals_list = np.array(eig_vals_list)
        colors = ["r" if eig_vals_list[i,1]>1 else "b" for i in range(eig_vals_list.shape[0])]
        plt.scatter(eig_vals_list[:,0], eig_vals_list[:,1], c=colors, s=1)
        plt.plot([0,1], [1,1], linestyle="--", color="r")
        plt.plot([1,1], [0,1], linestyle="--", color="r")
        plt.xlabel(r"Second-largest modulus of"+"\n"+r"eigenvalues of $P_{\bar{\pi}^*}$", fontsize=22)
        # ax = plt.gca()
        # ax.xaxis.set_label_coords(0.41, -0.12)
        plt.ylabel(r"Spectral radius of $\Phi$", fontsize=22)
        plt.xticks(fontsize=22, ticks=np.arange(0,1.2,0.2))
        plt.yticks(fontsize=22, ticks=np.arange(0,2.5,0.5))
        plt.xlim([-0.02, 1.1])
        plt.ylim([-0.02, 2])
        # plt.title("{}, unichain thresh {}, unstable / unichain = {} / {}".format(distr_and_parameter, unichain_threshold, unstable_unichain_count, unichain_count))
        plt.tight_layout()
        plt.savefig("figs2/eigen-scatter-{}-size-{}.pdf".format(distr_and_parameter, sspa_size))
        # plt.savefig("formal_figs/eigen-{}.pdf".format(distr_and_parameter))
        plt.show()


    ## ---save all examples up to specified i---
    if save_first_i_examples_unselected > 0:
        for i, setting in enumerate(all_data["examples"]):
            if setting is None:
                continue
            # if np.any(setting.y[:,0]+setting.y[:,1] < 1e-7):
            #     # skip the ambiguous examples where the definition of local stability is not unique
            #     continue
            if i >= save_first_i_examples_unselected:
                break
            os.makedirs("setting_data/unselected", exist_ok=True)
            setting_save_path =  "setting_data/unselected/random-size-{}-{}-({})".format(sspa_size, distr_and_parameter, i)
            print("saving the example {} to {}".format(i, setting_save_path))
            if os.path.exists(setting_save_path):
                print(setting_save_path+" exists!")
            else:
                save_bandit(setting, setting_save_path, {"alpha":alpha})


    ## ---simulating unstable examples---
    if do_simulation:
        print("Simulation starts")
        for policy_name in policy_names:
            for i, setting in enumerate(all_data["examples"]):
                if i >= simulate_up_to_ith:
                    break
                if (setting is None) or (setting.local_stab_eigval <= 1) or (setting.unichain_eigval >= 1):
                    # only simulate the locally unstable and unichain examples
                    continue
                print("Simulating the {}-th setting, policy={}".format(i, policy_name))
                # increase the lists simu_data[policy_name] and simu_data[(policy_name+"-final-state")] if not long enough
                while i >= len(simu_data[policy_name]):
                    simu_data[policy_name].append([])
                while i >= len(simu_data[(policy_name+"-final-state")]):
                    simu_data[(policy_name+"-final-state")].append(None)
                # set up the example
                setting = all_data["examples"][i]
                act_frac = setting.suggest_act_frac
                analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
                y = analyzer.solve_lp(verbose=False)[1]
                if policy_name == "lppriority":
                    priority_list = analyzer.solve_LP_Priority(verbose=False)
                elif policy_name == "whittle":
                    priority_list = analyzer.solve_whittles_policy()
                    if type(priority_list) is int:
                        # if non-indexable or multichain, just set the reward of whittle index to be zero
                        simu_data[policy_name][i] = [0 for j in range(simu_thousand_steps)]
                        continue
                else:
                    raise NotImplementedError
                policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)
                if len(simu_data[policy_name][i]) < simu_thousand_steps:
                    # restore the last state
                    if (simu_data[(policy_name+"-final-state")][i] is None) or (len(simu_data[policy_name][i]) == 0):
                        # no last state to restore, initialize new state
                        last_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
                    else:
                        last_states = simu_data[(policy_name+"-final-state")][i]
                    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=last_states)
                    # restore the current iteration number
                    cur_iter = len(simu_data[policy_name][i]) * 1000
                    # main simulation loop
                    total_reward_cur_thousand = 0
                    for t in range(cur_iter, simu_thousand_steps*1000):
                        actions = policy.get_actions(last_states)
                        instant_reward = rb.step(actions)
                        total_reward_cur_thousand += instant_reward
                        last_states = rb.get_states()
                        if (t+1)%1000 == 0:
                            # store the average reward every 1000 steps
                            simu_data[policy_name][i].append(total_reward_cur_thousand / 1000)
                            total_reward_cur_thousand = 0
                    # store the final state
                    simu_data[(policy_name+"-final-state")][i] = last_states
                print("setting id={}, eig P={}, eig Phi={}, upper bound={:0.4f}, lpp-limit-reward={:0.4f}, whittle-limit-reward={:0.4f}, ".format(
                    i, setting.unichain_eigval, setting.local_stab_eigval, setting.avg_reward_upper_bound,
                    setting.avg_reward_lpp_mf_limit, setting.avg_reward_whittle_mf_limit),
                    end=""
                )
                print("policy name = {}, avg reward={:0.4f}, std={:0.4f}".format(
                    policy_name,
                    sum(simu_data[policy_name][i])/simu_thousand_steps,
                    np.std(np.array(simu_data[policy_name][i])) / np.sqrt(simu_thousand_steps))
                )

    ## ---calculating subopt ratios for unstable examples---
    if plot_subopt_cdf or save_mix_examples or save_subopt_examples:
        subopt_ratios = []
        subopt_ratios_w = []
        for i, setting in enumerate(all_data["examples"]):
            if setting is None:
                continue
            if np.any(setting.y[:,0]+setting.y[:,1] < 1e-7):
                # skip the ambiguous examples where the definition of local stability is not unique
                continue
            if i >= simulate_up_to_ith:
                break
            if (setting.local_stab_eigval > 1) and (setting.unichain_eigval < 1):
                # print("the {}-th example is locally unstable".format(i))
                if np.any(setting.y[:,1]+setting.y[:,0]<1e-7):
                    print(setting.y[:,1]+setting.y[:,0])
                # subopt_ratio = setting.avg_reward_lpp_mf_limit / setting.avg_reward_upper_bound
                # subopt_ratio_w = setting.avg_reward_whittle_mf_limit / setting.avg_reward_upper_bound
                subopt_ratio = np.average(np.array(simu_data["lppriority"][i])) / setting.avg_reward_upper_bound
                subopt_ratio_w = np.average(np.array(simu_data["whittle"][i])) / setting.avg_reward_upper_bound
                subopt_ratios.append(subopt_ratio)
                subopt_ratios_w.append(subopt_ratio_w)
                if (subopt_ratio < 0.9) and (subopt_ratio_w < 0.9):
                    # print("In the {}-th example, lp index is {}-optimal, Whittle index is {}-suboptimal, rho(Phi)={}".format(i, subopt_ratio, subopt_ratio_w, setting.local_stab_eigval))
                    os.makedirs("setting_data/local_unstable_subopt", exist_ok=True)
                    setting_save_path =  "setting_data/local_unstable_subopt/random-size-{}-{}-({})".format(sspa_size, distr_and_parameter, i)
                    ## save suboptimal examples
                    if save_subopt_examples:
                        print("saving the example...")
                        if os.path.exists(setting_save_path):
                            print(setting_save_path+" exists!")
                        else:
                            save_bandit(setting, setting_save_path, {"alpha":alpha})
                    ## find a nearly unstable and suboptimal example and save its interpolation with a stable example.
                    if save_mix_examples and (setting.local_stab_eigval < 1.05):
                        print("In the {}-th example, lp index is {}-optimal, Whittle index is {}-suboptimal, rho(Phi)={}".format(i, subopt_ratio, subopt_ratio_w, setting.local_stab_eigval))
                        stable_setting_index = 2270
                        stable_setting = all_data["examples"][stable_setting_index]
                        keep_ratio = 0.95
                        new_setting = rb_settings.RandomExample(sspa_size, distr, parameters=[alpha])
                        new_setting.distr = "mix"
                        new_setting.trans_tensor = keep_ratio*setting.trans_tensor + (1-keep_ratio)*stable_setting.trans_tensor
                        new_setting.reward_tensor = keep_ratio*setting.reward_tensor + (1-keep_ratio)*stable_setting.reward_tensor
                        new_setting.suggest_act_frac = keep_ratio*setting.suggest_act_frac + (1-keep_ratio)*stable_setting.suggest_act_frac
                        new_setting.suggest_act_frac = int(100*new_setting.suggest_act_frac) / 100
                        result = compute_P_and_Phi_eigvals(new_setting)
                        if result is None:
                            print("{}({})+{}({}) has ambiguous stability".format(keep_ratio, i, 1-keep_ratio, stable_setting_index))
                        else:
                            analyzer = result[0]
                            new_setting.avg_reward_upper_bound = analyzer.opt_value
                            new_setting.y = analyzer.y.copy
                            new_setting.lp_priority = analyzer.solve_LP_Priority(verbose=False)
                            new_setting.whittle_priority = analyzer.solve_whittles_policy()
                            new_setting.unichain_eigval = result[1]
                            new_setting.local_stab_eigval = result[2]
                            print("{}({})+{}({}): upper bound={}, unichain_eigval={}, rho(Phi)={}".format(keep_ratio, i, 1-keep_ratio, stable_setting_index,
                                                            new_setting.avg_reward_upper_bound, new_setting.unichain_eigval, new_setting.local_stab_eigval))
                            setting_save_path = "setting_data/mix-random-size-{}-{}-({})-({})-ratio-{}".format(sspa_size, distr_and_parameter, i, stable_setting_index, keep_ratio)
                            if (new_setting.local_stab_eigval < 1) and (new_setting.local_stab_eigval > 0.9):
                                save_bandit(new_setting, setting_save_path, None)
        print("{} examples are locally unstable".format(len(subopt_ratios)))


    ## ---visualize the CDF of suboptimality among locally unstable examples---
    if plot_subopt_cdf:
        name_data_dict = {"lpp":subopt_ratios, "whittle":subopt_ratios_w,
                          "max":[max(subopt_ratios[i], subopt_ratios_w[i]) for i in range(len(subopt_ratios))]}
        for name, data in name_data_dict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.8)
            data.append(0)
            data = np.sort(data)
            print(bisect(data, 0.9) / len(data), bisect(data, 0.95) / len(data))
            ax.plot(data, np.linspace(0, 1, len(subopt_ratios)), color="b")
            ax.grid()
            # ax.hist(subopt_ratios, bins=20, weights=np.ones(len(subopt_ratios)) / len(subopt_ratios))
            if name == "lpp":
                full_name = "LP index"
            elif name == "whittle":
                full_name = "Whittle index"
            elif name == "max":
                full_name = "max of two indices"
            # title=ax.set_title("Size-{}-{} \n Subopt of {}'s avg reward when N={} among {} non-UGAP examples\n ".format(sspa_size, distr_and_parameter, full_name, N, len(subopt_ratios)))
            # title.set_y(1.05)
            plt.ylabel("CDF", fontsize=22)
            plt.xlabel("Optimality ratio", fontsize=22)
            plt.xticks(fontsize=22, ticks=np.arange(0, 1.2, step=0.2))
            plt.yticks(fontsize=22)
            fig.tight_layout()
            # plt.savefig("formal_figs/nonugap-subopt-{}.pdf".format(name))
            plt.savefig("figs2/nonugap-subopt-{}-size-{}-{}.pdf".format(name, sspa_size, distr_and_parameter))
            plt.show()


    ## ---simulating FTVA for all examples---
    if do_ftva_simulation:
        print("Simulation of FTVA starts")
        for i, setting in enumerate(all_data["examples"]):
            if i >= simulate_ftva_up_to_ith:
                break
            if setting is None:
                continue
            # simulate for all examples
            print("Simulating the {}-th setting, policy=FTVA".format(i))
            tic = time.time()
            # set up the example
            setting = all_data["examples"][i]
            act_frac = setting.suggest_act_frac
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
            y = analyzer.solve_lp(verbose=False)[1]
            if len(simu_ftva_data["FTVA"][i]) < simu_thousand_steps:
                # restore the final state
                if (simu_ftva_data["FTVA-final-state"][i] is None) or (len(simu_ftva_data["FTVA"][i]) == 0):
                    last_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
                    policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N,
                                act_frac=act_frac, init_virtual=None)
                else:
                    last_states, last_virtual_states = simu_ftva_data["FTVA-final-state"][i]
                    policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N,
                                act_frac=act_frac, init_virtual=last_virtual_states)
                rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=last_states)
                # restore the current iteration number
                cur_iter = len(simu_ftva_data["FTVA"][i]) * 1000
                # main simulation loop
                total_reward_cur_thousand = 0
                for t in range(cur_iter, simu_thousand_steps*1000):
                    actions, virtual_actions = policy.get_actions(last_states)
                    instant_reward = rb.step(actions)
                    total_reward_cur_thousand += instant_reward
                    new_states = rb.get_states()
                    policy.virtual_step(last_states, new_states, actions, virtual_actions)
                    last_states = new_states.copy()
                    if (t+1)%1000 == 0:
                        # store the average reward every 1000 steps
                        simu_ftva_data["FTVA"][i].append(total_reward_cur_thousand / 1000)
                        total_reward_cur_thousand = 0
                # store the final state
                simu_ftva_data["FTVA-final-state"][i] = (last_states, policy.virtual_states.copy())
            print("setting id={}, eig P={:0.4f}, SA hitting={:0.4f}, upper bound={:0.4f}, ".format(
                i, setting.unichain_eigval, setting.sa_max_hitting_time, setting.avg_reward_upper_bound),
                end=""
            )
            print("ftva avg reward={:0.4f}, std={:0.4f}, opt ratio={:0.2f}%; time={}".format(
                sum(simu_ftva_data["FTVA"][i])/simu_thousand_steps,
                np.std(np.array(simu_ftva_data["FTVA"][i])) / np.sqrt(simu_thousand_steps),
                100*sum(simu_ftva_data["FTVA"][i])/simu_thousand_steps/setting.avg_reward_upper_bound,
                time.time()-tic)
            )


    ## ---simulating mean-field dynamics of locally stable examples to find non-UGAP---
    if do_local_stable_simulation:
        print("Mean-field simulation of locally stable example starts. " +
              "{} initial points for each example".format(num_inits_for_test_UGAP))
        for policy_name in policy_names:
            for i, setting in enumerate(all_data["examples"]):
                if i >= simulate_local_stable_up_to_ith:
                    break
                if (setting is None) or (setting.local_stab_eigval > 1):
                    # only simulate the locally STABLE examples
                    continue
                print("Simulating the mean-field of {}-th setting, policy={}".format(i, policy_name))
                # set up the example
                setting = all_data["examples"][i]
                act_frac = setting.suggest_act_frac
                analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
                y = analyzer.solve_lp(verbose=False)[1]
                if policy_name == "lppriority":
                    priority_list = analyzer.solve_LP_Priority(verbose=False)
                elif policy_name == "whittle":
                    priority_list = analyzer.solve_whittles_policy()
                    if type(priority_list) is int:
                        simu_data[policy_name][i] = [0 for i in range(simu_thousand_steps)]
                        continue
                else:
                    raise NotImplementedError
                existing_inits = len(simu_local_stable_data[policy_name][i])
                for init_index in range(num_inits_for_test_UGAP):
                    init_state_fracs = np.random.dirichlet([1]*sspa_size)
                    if init_index < existing_inits:
                        continue
                    mf_reward = compute_mf_reward(setting, init_state_fracs, priority_list, T_mf_simulation)
                    simu_local_stable_data[policy_name][i].append(mf_reward)
                    if mf_reward / setting.avg_reward_upper_bound < 0.99:
                        print("setting id={}, policy={}, local stability eig={:0.4f}, unichain eig={:0.4f}, upper bound={:0.4f}, mf_reward={:0.4f}, opt ratio={:0.4f}".format(
                            i, policy_name, setting.local_stab_eigval, setting.unichain_eigval, setting.avg_reward_upper_bound, mf_reward, mf_reward / setting.avg_reward_upper_bound))
        ## count the number of locally-stable non-UGAP instances
        for policy_name in policy_names:
            num_non_UGAP_local_stable = 0
            num_mean_field_simulated = 0
            for i, setting in enumerate(all_data["examples"]):
                if i >= simulate_local_stable_up_to_ith:
                    break
                if (setting is None) or (setting.local_stab_eigval > 1):
                    # only show the locally STABLE examples
                    continue
                worst_opt_ratio = np.min(np.array(simu_local_stable_data[policy_name][i]) / setting.avg_reward_upper_bound) if len(simu_local_stable_data[policy_name][i])>0 else 1
                if worst_opt_ratio < UGAP_subopt_threshold:
                    print("Example {} is locally stable but {}-non-UGAP, with worst_opt_ratio={}".format(i, policy_name, worst_opt_ratio))
                    num_non_UGAP_local_stable += 1
                if len(simu_local_stable_data[policy_name][i]) > 0:
                    num_mean_field_simulated += 1
            print("For policy {}, there are {} locally stable non-UGAP instances among {} random instances that has been simulated".format(
                policy_name, num_non_UGAP_local_stable, num_mean_field_simulated))


    ## ---plot CDF of log hitting time in the SA assumption---
    if plot_sa_hitting_time:
        all_hitting_times = []
        for i, setting in enumerate(all_data["examples"]):
            if setting is not None:
                all_hitting_times.append(np.log10(setting.sa_max_hitting_time))
        all_hitting_times = sorted(all_hitting_times)
        plt.plot(all_hitting_times, np.linspace(0, 1, len(all_hitting_times)))
        plt.plot([0, 3.5], [1,1], linestyle="--")
        plt.plot([0, 3.5], [0,0], linestyle="--")
        # plt.hist(all_hitting_times)
        plt.xlabel("log_10 of max hitting times of leader-follower system")
        plt.ylabel("cdf")
        plt.xlim([0, 3.5])
        plt.grid()
        plt.title("CDF of log_10(Max hitting time) in {} {} examples".format(num_examples, distr_and_parameter))
        plt.savefig("figs2/sa-hit-time-log10-size-{}-{}.png".format(sspa_size, distr_and_parameter))
        plt.show()


    ## ---plot hitting time versus optimality ratio---
    if plot_sa_hitting_time_vs_opt:
        plot_hitting_times = []
        plot_opt_gap_ratio = []
        for i, setting in enumerate(all_data["examples"]):
            if (setting is None) or (i >= simulate_ftva_up_to_ith):
                continue
            plot_hitting_times.append(setting.sa_max_hitting_time)
            plot_opt_gap_ratio.append(1 - sum(simu_ftva_data["FTVA"][i]) / (simu_thousand_steps*setting.avg_reward_upper_bound))
        plt.scatter(plot_hitting_times, plot_opt_gap_ratio, s=1)
        plt.xlabel("max hitting times of leader-follower system")
        plt.ylabel("opt gap ratio of FTVA")
        plt.title("max hitting time v.s. FTVA opt gap ratio in {} {} examples".format(num_examples, distr_and_parameter))
        plt.savefig("figs2/sa-hit-time-ftva-reward-size-{}-scatter-{}.png".format(sspa_size, distr_and_parameter))
        plt.show()

    # "almost unstable instances", not included in the paper
    if find_almost_unstable_examples:
        Phi_radiuses = -np.ones((num_examples,))
        nondegeneracies = -np.ones((num_examples,))
        for i,setting in enumerate(all_data["examples"]):
            if (setting is None) or (setting.local_stab_eigval >= 1) or (setting.unichain_eigval >= 1):
                # only consider locally stable examples
                continue
            if np.any(setting.y[:,0]+setting.y[:,1] < 1e-7):
                # skip the ambiguous examples where the definition of local stability is not unique
                continue
            nondegeneracies[i] = y2nondegeneracy(setting.y)
            if nondegeneracies[i] > 0.15:
                Phi_radiuses[i] = setting.local_stab_eigval
        sorted_indices = np.flip(np.argsort(Phi_radiuses))
        sorted_Phi_radiuses = np.flip(np.sort(Phi_radiuses))
        # print(np.sum(Phi_radiuses>0))
        # indices_to_save = np.sort(np.random.choice(np.arange(0,350), size=10))
        target_radiuses = [0.99, 0.95, 0.90]
        num_to_save_for_each_target = 2
        rem_num = num_to_save_for_each_target
        for j in range(350):
            if Phi_radiuses[sorted_indices[j]] < target_radiuses[0]:
                if rem_num > 0:
                    print("index={}, rho(Phi)={}, nondegeneracy={}, \n {}".format(sorted_indices[j], Phi_radiuses[sorted_indices[j]],
                                                                               nondegeneracies[sorted_indices[j]], all_data["examples"][sorted_indices[j]].y))
                    setting_save_path = "setting_data/stable-size-{}-{}-({})".format(sspa_size, distr_and_parameter, sorted_indices[j])
                    if save_almost_unstable_examples:
                        print("saving the example...")
                        if os.path.exists(setting_save_path):
                            print(setting_save_path+" exists!")
                        else:
                            save_bandit(all_data["examples"][sorted_indices[j]], setting_save_path, {"alpha":alpha})
                            print(setting_save_path+" saved!")
                    rem_num -= 1
                else:
                    target_radiuses.pop(0)
                    rem_num = num_to_save_for_each_target
            if len(target_radiuses) == 0:
                break


    ## ----- save data -----
    if update_database:
        print("saving data... do not quit...")
        with open(file_path, "wb") as f:
            pickle.dump(all_data, f)
        with open(simu_file_path, "wb") as f:
            pickle.dump(simu_data, f)
        with open(simu_ftva_file_path, "wb") as f:
            pickle.dump(simu_ftva_data, f)
        with open(simu_local_stable_path, "wb") as f:
            pickle.dump(simu_local_stable_data, f)
    else:
        print("no-update mode, example data not updated.")
    print("Finished!")


if __name__ == "__main__":
    np.random.seed(114514)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    ## Script Execution Instructions
    # --------------------------------------------------------------------------
    # This script should be run multiple times with different input parameters
    # to generate the data and figures for the paper.
    #
    # To reproduce specific figures, use the following configurations:
    #
    #   - Figure 7(a):      `input_alpha=1.0`,  `input_plot_cdf=False`, `input_distr="dirichlet"`
    #   - Figure 7(b):      `input_alpha=0.2`,  `input_plot_cdf=False`, `input_distr="dirichlet"`
    #   - Figures 7(c), 8:  `input_alpha=0.05`, `input_plot_cdf=True`,  `input_distr="dirichlet"`
    #   - Figure 13(a):     `input_alpha=0.2`,  `input_plot_cdf=False`, `input_distr="dirichlet-uniform-nzR0"`
    #   - Figures 13(b), 14:`input_alpha=0.05`, `input_plot_cdf=True`,  `input_distr="dirichlet-uniform-nzR0"`
    #
    # NOTE:
    # - A full run with default parameters is time-consuming. For quicker tests,
    #   you can reduce `num_examples` (e.g., to 5000) and `simulate_up_to_ith`
    #   (e.g., to 1000) within the function.
    # - The script saves progress at the end of the run, so you can increase these
    #   values incrementally across runs without losing previous work.
    # --------------------------------------------------------------------------
    search_and_store_unstable_examples(input_alpha=0.05, input_plot_cdf=True, input_distr="dirichlet")
