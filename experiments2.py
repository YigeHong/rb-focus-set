#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Primary simulation script for evaluating policies in various Restless
Bandit (RB) instances.

This script runs the core experiments for the paper "Unichain and Aperiodicity
are Sufficient for Asymptotic Optimality of Restless Bandits" (hereafter
referred to as "RB-unichain").
"""

__author__ = "Yige Hong"
__date__ = "2025-09-06"
__version__ = "1.0"


import numpy as np
import cvxpy as cp
import scipy

from discrete_RB import *
import rb_settings
import time
import pickle
from matplotlib import pyplot as plt
import os
import multiprocessing as mp
import bisect


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, Ns=None, save_dir="fig_data",
                 skip_N_below=None, no_run=False, debug=False, note=None, init_state_fracs=None):
    """
    The main simulation function.
    Args:
        setting_name (str): The name of the Restless Bandit (RB) instance.
            This name is used to instantiate the correct RB object from
            "rb_setting.py" and to name simulation data files.

            If `setting_path` is None, this name defines the instance.
            Choices include:
            - "eight-states", "three-states", "non-sa"
            - "eight-states-045", "new-eight-states"
            - "new2-eight-states", "new2-eight-states-045"
            - "non-sa-big1", "non-sa-big2", "non-sa-big3", "non-sa-big4"
            - f"conveyor-belt-nd-{sspa_size}"

            If `setting_path` is provided, the RB instance is loaded from the
            path, and this name is only used for simulation data file naming.

            Notes on specific instances:
            - "three states" & "new2-eight-states": Counterexamples for UGAP
              (Section 8.1 of the RB-unichain paper).
            - "non-sa" & "non-sa-big2": Counterexamples for SA
              (Section 8.3 of the RB-unichain paper).

        policy_name (str): The policy to be simulated. This name is used to
            instantiate the policy object from "discrete_RB.py" and to name
            simulation data files.

            Choices include:
            - "id": ID policy
            - "setexp": Set-expansion policy
            - "setexp-priority": Set-expansion policy with LP index
            - "setopt": Set-optimization policy (Appendix F of RB-unichain paper)
            - "whittle": Whittle index policy
            - "lppriority": LP index policy
            - "ftva": FTVA policy from the paper "Restless Bandits with
              Average Reward: Breaking the Uniform Global Attractor Assumption".
            - "random-tb": Samples actions using the optimal single-armed
              policy, then uses random tie-breaking.
            - "twoset-v1", "twoset-integer", "twoset-faithful": Policies
              from a followup paper, not simulated in the paper RB-unichain
            - "setopt-tight", "setexp-id", "setopt-id", "setopt-priority":
              Customized policies not appearing in the paper.

        init_method (str): The method for initializing the RB's state distribution.
            Choices include:
            - "random": Samples the state of each arm independently and
              uniformly at random.
            - "same": Sets the initial state of all arms to 0.
            - "given": Initializes arms based on specified fractions in
              the `init_state_fracs` parameter.
            - "bad": A specific initialization for "eight-states" instances
              designed to make the LP index perform poorly (not used in
              the RB-unichain paper's experiments).

       T (int): The simulation horizon.

        setting_path (str, optional): The path to a .pkl file containing a
            `RandomExample` object (defined in `rb_setting.py`). This is used
            for loading randomly generated RB instances. Defaults to None.

        Ns (list[int], optional): A list of arm counts (N) for which to run
            the simulation. If None, a default list of values will be used.

        save_dir (str): The directory where simulation data will be saved.

        skip_N_below (int, optional): If provided, skips simulations for any N
            in the `Ns` list that is below this threshold. Useful for resuming
            interrupted experiments. Defaults to None.

        no_run (bool): If True, the function will print the simulation
            settings without actually running the simulation. Defaults to False.

        debug (bool): If True, prints additional information during the
            simulation for debugging purposes, and do not save data. Defaults to False.

        note (str, optional): If provided, this string is appended to the
            saved data file's name (as f"_{note}") to prevent overwriting
            existing files. Defaults to None.

        init_state_fracs (np.ndarray, optional): A NumPy array specifying the
            initial fraction of arms in each state. This is only used when
            `init_method` is set to "given". Defaults to None.
    """
    # load the RB instance
    if setting_name == "eight-states":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    elif setting_name == "three-states":
        setting = rb_settings.Gast20Example2()
    elif setting_name == "non-sa":
        setting = rb_settings.NonSAExample()
    elif setting_name == "eight-states-045":
        probs_L, probs_R, action_script, suggest_act_frac, _ = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.suggest_act_frac = 0.45
    elif setting_name == "new-eight-states":
        probs_L, probs_R, action_script, suggest_act_frac, _ = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.02
    elif setting_name == "new2-eight-states":
        probs_L, probs_R, action_script, suggest_act_frac, _ = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.1/30
    elif setting_name == "new2-eight-states-045":
        probs_L, probs_R, action_script, suggest_act_frac, _ = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.1/30
        setting.suggest_act_frac = 0.45
    elif setting_name.startswith("conveyor-belt-nd"):
        sspa_size = int(setting_name.split("-")[-1])
        probs_L, probs_R, action_script, suggest_act_frac, r_disturb = rb_settings.ConveyorExample.get_parameters(
            "arbitrary-length", sspa_size, nonindexable=True)
        setting = rb_settings.ConveyorExample(sspa_size, probs_L, probs_R, action_script, suggest_act_frac, r_disturb=r_disturb)
    elif setting_name == "non-sa-big1":
        setting = rb_settings.BigNonSAExample("v1")
    elif setting_name == "non-sa-big2":
        setting = rb_settings.BigNonSAExample("v2")
    elif setting_name == "non-sa-big3":
        setting = rb_settings.BigNonSAExample("v3")
    elif setting_name == "non-sa-big4":
        setting = rb_settings.BigNonSAExample("v4")
    elif setting_path is not None:
        setting = rb_settings.ExampleFromFile(setting_path)
    else:
        raise NotImplementedError
    act_frac = setting.suggest_act_frac
    if Ns is None:
        Ns = list(range(100,1100,100))
    print("Ns = ", Ns)
    for N in Ns:
        # ensure alpha N is integer
        assert np.allclose(N*act_frac, round(N*act_frac), atol=1e-7), "N={}, act_frac={}, N*act_frac={}, round(N*act_frac)={}".format(
            N, act_frac, N*act_frac, round(N*act_frac))
    # WARNING: Do not change `num_reps`.
    # To simulate more repetitions, run this function multiple times, using the
    # `note` argument to create distinct output files for each run.
    num_reps = 1
    # NOTE: For data generated before September 28, 2024, `save_mean_every'
    # was not explicitly specified and should be regarded as 1.
    save_mean_every = 1000
    print()
    rb_settings.print_bandit(setting)

    # Preprocessing stage. Solve LP relaxation, Whittle index, LP index, and some other parameters.
    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    print("min(y(sneu,1), y(sneu,0)) = ", y2nondegeneracy(y))
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("2*lambda_W = ", 2*np.linalg.norm(W, ord=2))
    if setting_name == "eight-states":
        priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
    else:
        priority_list = analyzer.solve_LP_Priority()
    print("priority list =", priority_list)
    whittle_priority = analyzer.solve_whittles_policy()
    print("Whittle priority=", whittle_priority)
    U = analyzer.compute_U(abstol=1e-10)[0]
    if U is not np.inf:
        print("spectral norm of U=", np.max(np.abs(np.linalg.eigvals(U))))
    else:
        print("U diverges, locally unstable")

    if no_run:
        return

    # define file name, load existing data if there is any
    if note is None:
        data_file_name = "{}/{}-{}-N{}-{}-{}".format(save_dir, setting_name, policy_name, Ns[0], Ns[-1], init_method)
    else:
        data_file_name = "{}/{}-{}-N{}-{}-{}-{}".format(save_dir, setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
    if os.path.exists(data_file_name):
        # check the meta-data of the file that we will overwrite;
        # stop the simulation if there is inconsistency
        with open(data_file_name, "rb") as f:
            setting_and_data = pickle.load(f)
            assert setting_and_data["num_reps"] == num_reps
            assert setting_and_data["T"] == T
            assert setting_and_data["act_frac"] == act_frac
            assert setting_and_data["Ns"] == Ns
            assert setting_and_data["name"] == setting_name
            assert setting_and_data["policy_name"] == policy_name
            assert setting_and_data["setting_code"] == setting_name
            assert setting_and_data["init_method"] == init_method
            if save_mean_every in setting_and_data:
                assert setting_and_data["save_mean_every"] == save_mean_every
            else:
                assert save_mean_every == 1

    # start simulation loop
    tic = time.time()
    reward_array = np.nan * np.empty((num_reps, len(Ns)))
    full_reward_trace = {}
    full_ideal_acts_trace = {}
    for i, N in enumerate(Ns):
        if (skip_N_below is not None) and (N <= skip_N_below):
            continue
        # The index `i` in the dictionary key is redundant but maintained for
        # consistency with previously saved simulation data.
        full_reward_trace[i, N] = []
        # Stores the trace of the fraction of arms that take ideal actions under
        # the ID or set-expansion policy. This data is used for the analysis in
        # Appendix H.5 of the RB-unichain paper.
        full_ideal_acts_trace[i, N] = []
        # we always have num_reps = 1
        for rep in range(num_reps):
            # initialize the states
            if init_method == "random":
                init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
            elif init_method == "same":
                init_states = np.zeros((N,))
            elif init_method == "bad":
                init_states = np.random.choice(np.arange(round(setting.sspa_size/2), setting.sspa_size), N, replace=True)
            elif init_method == "given":
                init_states = np.zeros((N,))
                for s in range(setting.sspa_size):
                    start_ind = round(N * np.sum(init_state_fracs[0:s]))
                    end_ind = round(N * np.sum(init_state_fracs[0:(s+1)]))
                    init_states[start_ind: end_ind] = s
            else:
                raise NotImplementedError
            rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
            total_reward = 0
            conformity_count = 0
            non_shrink_count = 0
            focus_set = np.array([], dtype=int)
            # `OL_set` `EP_set` are for a followup paper and are not relevant to the RB-unichain paper
            OL_set = np.array([], dtype=int)
            EP_set = np.arange(0, max(int(min(act_frac,1-act_frac)*N)-1,0),dtype=int)

            # simulation loop for each policy
            if policy_name == "id":
                policy = IDPolicy(setting.sspa_size, y, N, act_frac)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    actions, num_ideal_acts = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
            elif policy_name == "setexp":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set, non_shrink_flag = policy.get_new_focus_set(cur_states=cur_states,
                                                                          last_focus_set=focus_set)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    non_shrink_count += non_shrink_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
                    #     print("focus set size before rounding={}, after rounding={}".format(policy.m.value*N, len(focus_set)))
                    #     print("conformity count = {}, non-shrink count = {}".format(conformity_count, non_shrink_count))
            elif policy_name == "setopt":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
                    #     print("focus set size before rounding={}, after rounding={}".format(policy.m.value*N, len(focus_set)))
                    #     print("conformity count = {}".format(conformity_count))
            elif policy_name == "setopt-tight":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states, subproblem="tight") ##
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "setexp-id":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set, non_shrink_flag = policy.get_new_focus_set(cur_states=cur_states,
                                                                          last_focus_set=focus_set)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="ID")
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    non_shrink_count += non_shrink_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "setopt-id":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="ID")
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "setexp-priority":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set, non_shrink_flag = policy.get_new_focus_set(cur_states=cur_states,
                                                                          last_focus_set=focus_set)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="priority",
                                                                  tb_priority=priority_list)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    non_shrink_count += non_shrink_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "setopt-priority":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="priority",
                                                                  tb_priority=priority_list)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "ftva":
                policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N,
                                    act_frac=act_frac, init_virtual=None)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    prev_state = rb.get_states()
                    actions, virtual_actions = policy.get_actions(prev_state)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    num_good_arms = np.sum(np.all([actions==virtual_actions, prev_state==policy.virtual_states], axis=0))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    new_state = rb.get_states()
                    policy.virtual_step(prev_state, new_state, actions, virtual_actions)
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(focus_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "lppriority":
                policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)
                recent_total_reward = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    if debug:
                        print(states_to_scaled_state_counts(setting.sspa_size, N, cur_states))
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        recent_total_reward = 0
            elif policy_name == "whittle":
                if whittle_priority is -1:
                    print("Non-indexable!!!")
                    return
                elif whittle_priority is -2:
                    print("Multichain!!!")
                    return
                else:
                    policy = PriorityPolicy(setting.sspa_size, whittle_priority, N=N, act_frac=act_frac)
                recent_total_reward = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        recent_total_reward = 0
            elif policy_name == "random-tb":
                policy = RandomTBPolicy(setting.sspa_size, y, N=N, act_frac=act_frac)
                recent_total_reward = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    if debug:
                        print(states_to_scaled_state_counts(setting.sspa_size, N, cur_states))
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        recent_total_reward = 0
            elif policy_name == "twoset-v1":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
                # print("eta=", policy.eta)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(OL_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "twoset-integer":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U, rounding="misocp")
                # print("eta=", policy.eta)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(OL_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            elif policy_name == "twoset-faithful":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
                # print("eta=", policy.eta)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set)
                    actions, EP_set = policy.get_actions_with_EP_set(cur_states=cur_states, cur_OL_set=OL_set, last_EP_set=EP_set)
                    assert np.sum(actions) == round(act_frac*N), \
                        "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    recent_total_reward += instant_reward
                    recent_total_ideal_acts += len(OL_set)
                    if (t+1)%save_mean_every == 0:
                        full_reward_trace[i,N].append(recent_total_reward / save_mean_every)
                        full_ideal_acts_trace[i,N].append(recent_total_ideal_acts / save_mean_every)
                        recent_total_reward = 0
                        recent_total_ideal_acts = 0
            else:
                raise NotImplementedError
            avg_reward = total_reward / T
            reward_array[rep, i] = avg_reward
            avg_idea_frac = np.sum(np.array(full_ideal_acts_trace[i,N])) / (T*N)
            print("setting={}, policy={}, N={}, rep_id={}, avg reward = {}, avg ideal frac ={}, total gap={}, note={}".format(
                setting_name, policy_name, N, rep, avg_reward, avg_idea_frac, N*(analyzer.opt_value-avg_reward), note))

            # save data if not in the debug mode
            if not debug:
                if os.path.exists(data_file_name):
                    # if file exists, overwrite the data in place
                    with open(data_file_name, 'rb') as f:
                        setting_and_data = pickle.load(f)
                        setting_and_data["reward_array"][rep, i] = avg_reward
                        setting_and_data["full_reward_trace"][(i,N)] = full_reward_trace[(i,N)].copy()
                        setting_and_data["full_ideal_acts_trace"][(i,N)] = full_ideal_acts_trace[(i,N)].copy()
                else:
                    # if file does not exist, create a new one
                    setting_and_data = {
                        "num_reps": num_reps,
                        "T": T,
                        "act_frac": act_frac,
                        "Ns": Ns,
                        "name": setting_name,
                        "policy_name": policy_name,
                        "setting_code": setting_name,
                        "init_method": init_method,
                        "init_state_fracs": init_state_fracs,
                        "setting": setting,
                        "reward_array": reward_array,
                        "full_reward_trace": full_reward_trace,
                        "full_ideal_acts_trace": full_ideal_acts_trace,
                        "y": y,
                        "W": W,
                        "upper bound": analyzer.opt_value,
                        "save_mean_every": save_mean_every
                    }
                with open(data_file_name, 'wb') as f:
                    pickle.dump(setting_and_data, f)
    print("time for running one policy with T={} and {} data points is {}".format(T, len(Ns), time.time()-tic))



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=600)

    # The code below generates the data for the main policy-comparison figures
    # in the RB-unichain paper (Figures 6, 9, and 10).
    # The resulting data is saved to the `fig_data/` directory, which can
    # then be used by `plot_figures.py` to reproduce the plots.
    #
    # NOTE: Other figures are generated by different scripts:
    # - Figures 7, 8, 13, 14 (scatter plots and CDFs) are generated by
    #   `understand_assumptions.py`.
    # - Appendix H.5 figures (ID and set-expansion analysis) are generated
    #   by `understand_id_and_set_expansion.py`.
    my_pool = mp.Pool(10)
    task_list = []

    # --------------------------------------------------------------------------
    # DATA FOR FIGURES 6 & 10
    # Simulate instances from Section 8.1 (UGAP counterexamples) and
    # Section 8.3 (SA counterexamples).
    # --------------------------------------------------------------------------
    for setting_name in ["three-states", "new2-eight-states", "non-sa", "non-sa-big2"]:
        for policy_name in ["id", "setexp", "setexp-priority", "ftva", "lppriority", "whittle"]:
            if (setting_name in ["non-sa", "non-sa-big2"]) and (policy_name == "ftva"):
                T = 16e4
                Tstr = "16e4"
            else:
                T = 2e4
                Tstr = "2e4"
            for rep_id in range(1, 6):
                task_list.append(my_pool.apply_async(run_policies,
                                kwds={"setting_name": setting_name,
                                      "policy_name": policy_name,
                                      "init_method": "random",
                                      "T": T,
                                      "Ns": np.arange(100, 1100, 100),
                                      "save_dir": "fig_data",
                                      "note": f"T{Tstr}r{rep_id}"}))

    # --------------------------------------------------------------------------
    # DATA FOR FIGURE 9
    # Simulate the randomly generated "local unstable suboptimal" instances from
    # Section 8.2. These specific instances (`random-size-10-dirichlet-0.05-(355)`
    # and `random-size-10-dirichlet-0.05-(582)`) are stored in the folder `setting_data`,
    # and can also be generated by running `understand_assumptions.py` with seed 114514.
    # --------------------------------------------------------------------------
    for i in [355, 582]:
        setting_name = "random-size-10-dirichlet-0.05-({})".format(i)
        setting_path = "setting_data/local_unstable_subopt/" + setting_name
        for policy_name in ["id", "setexp", "setexp-priority", "ftva", "lppriority", "whittle"]:
            for rep_id in range(1, 6):
                task_list.append(my_pool.apply_async(run_policies,
                                        kwds={"setting_name": setting_name,
                                              "policy_name": policy_name,
                                              "init_method": "random",
                                              "T": 2e4,
                                              "setting_path": setting_path,
                                              "Ns": np.arange(100, 1100, 100),
                                              "save_dir": "fig_data",
                                              "note": "T2e4r{}".format(rep_id)}))

    my_pool.close()
    my_pool.join()

