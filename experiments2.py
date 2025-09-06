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


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, Ns=None, save_dir="fig_data", skip_N_below=None, no_run=False, debug=False, note=None, init_state_fracs=None):
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
        Ns = list(range(100,1100,100)) #list(range(1100, 4100, 100)) # list(range(2100,4100,200)) #  #list(range(1500, 5500, 500)) # list(range(1000, 20000, 1000))
    print("Ns = ", Ns)
    for N in Ns:
        # ensure alpha N is integer
        assert np.allclose(N*act_frac, round(N*act_frac), atol=1e-7), "N={}, act_frac={}, N*act_frac={}, round(N*act_frac)={}".format(N, act_frac, N*act_frac, round(N*act_frac))
    num_reps = 1
    save_mean_every = 1000 # save_mean_every=1 for data generated before Sep 28, 2024
    print()
    rb_settings.print_bandit(setting)

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

    if note is None:
        data_file_name = "{}/{}-{}-N{}-{}-{}".format(save_dir, setting_name, policy_name, Ns[0], Ns[-1], init_method)
    else:
        data_file_name = "{}/{}-{}-N{}-{}-{}-{}".format(save_dir, setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
    if os.path.exists(data_file_name):
        # check the meta-data of the file that we want to overwrite
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

    tic = time.time()
    reward_array = np.nan * np.empty((num_reps, len(Ns)))
    full_reward_trace = {}
    full_ideal_acts_trace = {}
    for i, N in enumerate(Ns):
        if (skip_N_below is not None) and (N <= skip_N_below):
            continue
        full_reward_trace[i,N] = []
        full_ideal_acts_trace[i,N] = []
        for rep in range(num_reps):
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
                print("!!!!!!!!!!", states_to_scaled_state_counts(setting.sspa_size, N, init_states))
            else:
                raise NotImplementedError
            rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
            total_reward = 0
            conformity_count = 0
            non_shrink_count = 0
            focus_set = np.array([], dtype=int)
            OL_set = np.array([], dtype=int)
            EP_set = np.arange(0, max(int(min(act_frac,1-act_frac)*N)-1,0),dtype=int)

            if policy_name == "id":
                policy = IDPolicy(setting.sspa_size, y, N, act_frac)
                recent_total_reward = 0
                recent_total_ideal_acts = 0
                for t in range(T):
                    cur_states = rb.get_states()
                    actions, num_ideal_acts = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
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
            print("setting={}, policy={}, N={}, rep_id={}, avg reward = {}, avg ideal frac ={}, total gap={}, note={}".format(setting_name, policy_name, N, rep,
                                                                                   avg_reward, avg_idea_frac, N*(analyzer.opt_value-avg_reward), note))

            if not debug:
                if os.path.exists(data_file_name):
                    # overwrite the data in place
                    with open(data_file_name, 'rb') as f:
                        setting_and_data = pickle.load(f)
                        setting_and_data["reward_array"][rep, i] = avg_reward
                        setting_and_data["full_reward_trace"][(i,N)] = full_reward_trace[(i,N)].copy()
                        setting_and_data["full_ideal_acts_trace"][(i,N)] = full_ideal_acts_trace[(i,N)].copy()
                else:
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

    my_pool = mp.Pool(10)

    task_list = []
    Ns = list(range(100,1100,100))
    setting_names = ["non-sa", "non-sa-big2"]
    save_dir = "fig_data"
    init_method = "given"
    for setting_name in setting_names:
        sspa_size = 8 if setting_name == "non-sa" else 12
        temp_init_state = np.random.choice(np.arange(0, sspa_size), 100, replace=True)
        init_state_fracs = states_to_scaled_state_counts(sspa_size, 100, temp_init_state)
        for rep in range(5):
            # setting_path = "setting_data/unselected/" + setting_name
            # if os.path.exists(setting_path):
            #     setting = rb_settings.ExampleFromFile(setting_path)
            # else:
            #     print("{} not found!!".format(setting_path))
            #     continue
            for policy_name in ["ftva"]:
                T = 160000
                note = "binom_init_T16e4r{}".format(rep)
                cur_save_path = "{}/{}-{}-N{}-{}-{}-{}".format(save_dir, setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
                if (not os.path.exists(cur_save_path)):# and (not (policy_name == "whittle")):
                    print(cur_save_path, "is not simulated yet")
                    task_list.append(
                        my_pool.apply_async(run_policies, args=(setting_name, policy_name, init_method, T, None, Ns),
                                            kwds={"save_dir": save_dir, "note": note, "init_state_fracs": init_state_fracs})
                    )
                    # run_policies(setting_name, policy_name, init_method, T, None, Ns, save_dir=save_dir, note=note, init_state_fracs=init_state_fracs)

    my_pool.close()
    my_pool.join()

