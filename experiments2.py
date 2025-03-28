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


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, Ns=None, save_dir="fig_data", skip_N_below=None, no_run=False, debug=False, note=None):
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
        assert np.allclose(N*act_frac, round(N*act_frac), atol=1e-7), "N={}, act_frac={}, N*act_frac={}".format(N, act_frac, N*act_frac)
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
                        "init_state_fracs": None,
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



def figure_from_multiple_files():
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["random-size-3-uniform-({})".format(i) for i in range(5)]
    settings = ["random-size-10-dirichlet-0.05-({})".format(i) for i in [355]]
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id"]
    skip_policies = []# ["setexp", "setexp-priority"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-"]
    policy_markers = ["v",".","^","s","p","*"]
    policy_colors = ["m","c","y","r","g","b"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)",
                    "twoset-v1":"Two set v1"}
    setting2legend_position = {"random-size-10-dirichlet-0.05-(582)": (1,0.1), "random-size-10-dirichlet-0.05-(355)":(1,0.1),
                               "new2-eight-states":"center right", "three-states":(1,0.1), "non-sa":"lower right", "non-sa-big2":"center right"}
    setting2yrange = {"random-size-10-dirichlet-0.05-(582)": None, "random-size-10-dirichlet-0.05-(355)":None,
                      "new2-eight-states":None, "three-states":None, "non-sa":(0.7,1.025), "non-sa-big2":None}
    batch_means_dict = {}
    target_num_batches = 20
    Ns = np.array(list(range(100,1100,100))) #np.array(list(range(1500, 5500, 500))) # list(range(1000, 20000, 1000))
    init_method = "random"

    for setting_name in settings:
        for policy_name in policies:
            file_prefix = "{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
            file_dir = "fig_data"
            if setting_name in ["non-sa", "non-sa-big1", "non-sa-big2", "non-sa-big3", "non-sa-big4"] and (policy_name == "ftva"):
                file_prefix += "-T16e4"
            else:
                file_prefix += "-T2e4"
            file_names = [file_name for file_name in os.listdir(file_dir) if file_name.startswith(file_prefix)]
            print("{}:{}".format(file_prefix, file_names))
            # if note is not None:
            #     file_name_alter = "fig_data/{}-{}-N{}-{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
            #     if os.path.exists(file_name_alter):
            #         file_name = file_name_alter
            if len(file_names) == 0:
                if policy_name == "whittle":
                    continue
                else:
                    raise FileNotFoundError("no file with prefix {}".format(file_prefix))
            batch_means_dict[(setting_name, policy_name)] = [[] for i in range(len(Ns))] # shape len(Ns)*num_batches
            for file_name in file_names:
                with open(os.path.join(file_dir, file_name), 'rb') as f:
                    setting_and_data = pickle.load(f)
                    full_reward_trace = setting_and_data["full_reward_trace"]
                    for i,N in enumerate(Ns):
                        print("time horizon of {} = {}".format(file_name, len(full_reward_trace[i,N])))
                        cur_batch_size = int(len(full_reward_trace[i,N])/4)
                        for t in range(0, len(full_reward_trace[i,N]), cur_batch_size):
                            batch_means_dict[(setting_name, policy_name)][i].append(np.mean(full_reward_trace[i, N][t:(t+cur_batch_size)]))
            for i in range(len(Ns)):
                assert len(batch_means_dict[(setting_name, policy_name)][i]) == target_num_batches
            batch_means_dict[(setting_name, policy_name)] = np.array(batch_means_dict[(setting_name, policy_name)])
            print(setting_name, policy_name, np.mean(batch_means_dict[(setting_name, policy_name)], axis=1), np.std(batch_means_dict[(setting_name, policy_name)], axis=1))

    for setting_name in settings:
        if setting_name == "eight-states":
            upper_bound = 0.0125
        elif setting_name == "three-states":
            upper_bound = 0.12380016733626052
        elif setting_name == "non-sa":
            upper_bound = 1
        else:
            files_w_prefix = [filename for filename in os.listdir("fig_data")
                              if filename.startswith("{}-{}-N{}-{}-{}".format(setting_name, "ftva", 100, 1000, init_method))]
            with open("fig_data/"+files_w_prefix[0], 'rb') as f:
                setting_and_data = pickle.load(f)
                upper_bound = setting_and_data["upper bound"]
        plt.plot(Ns, np.array([1] * len(Ns)), label="Upper bound", linestyle="--", color="k")
        for i,policy_name in enumerate(policies):
            if (policy_name == "whittle") and ((setting_name, policy_name) not in batch_means_dict):
                pass
            elif policy_name in skip_policies:
                pass
            else:
                cur_batch_means = batch_means_dict[(setting_name, policy_name)]
                plt.errorbar(Ns, np.mean(cur_batch_means, axis=1) / upper_bound,
                             yerr=2*np.std(cur_batch_means, axis=1)/np.sqrt(len(cur_batch_means)) / upper_bound,
                             label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                             marker=policy_markers[i], markersize=8, color=policy_colors[i])
            plt.xlabel("N", fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel("Optimality ratio", fontsize=14)
        plt.yticks(fontsize=14)
        if setting2yrange[setting_name] is not None:
            plt.ylim(setting2yrange[setting_name])
        plt.tight_layout()
        plt.grid()
        if setting2legend_position[setting_name] is None:
            plt.legend(fontsize=14)
        elif type(setting2legend_position[setting_name]) is str:
            plt.legend(fontsize=14,loc=setting2legend_position[setting_name])
        else:
            plt.legend(fontsize=14,loc="lower right", bbox_to_anchor=setting2legend_position[setting_name])
        plt.savefig("figs2/{}-N{}-{}-{}.pdf".format(setting_name, Ns[0], Ns[-1], init_method))
        plt.savefig("formal_figs/{}-{}-init.pdf".format(setting_name, init_method))
        plt.show()


def figure_from_multiple_files_flexible_N(note=None):
    """
    Plotting function that reads data files with Ns to combine into one plot.
    """
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["new2-eight-states-045"]+["conveyor-belt-nd-12"]+["random-size-8-uniform-({})".format(i) for i in [1, 6, 0]]
    # other possible settings ["random-size-8-uniform-({})".format(i) for i in [1]] # ["mix-random-size-10-dirichlet-0.05-({})-(2270)-ratio-0.95".format(i) for i in [1436, 6265]] #["stable-size-10-dirichlet-0.05-({})".format(i) for i in [4339]]#, 4149, 4116, 2667, 2270, 9632]]
    settings = ["new2-eight-states-045"]
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id", "twoset-v1", "twoset-faithful"]
    skip_policies =  ["setexp", "setexp-priority","twoset-v1"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-", "-", "-."]
    policy_markers = ["v",".","^","s","p","*", "v", "P"]
    policy_colors = ["m","c","y","r","g","b", "brown", "orange"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)",
                    "twoset-v1":"Two-set v1", "twoset-faithful":"Two-set policy"}
    setting2truncate = {"random-size-8-uniform-(0)": 10**(-6), "random-size-8-uniform-(1)": 10**(-7),
                        "random-size-8-uniform-(2)": 10**(-6), "random-size-8-uniform-(3)": 10**(-6)}
    truncate_level_default = 10**(-7)
    plot_CI = True
    batch_size_mode = "adaptive" #"fixed" or "adaptive"
    batch_size = 8000 # only if batch_size_mode = "fixed"
    burn_in_batch = 1
    # mode = "fixed" # "fixed", or "range"
    N_range = [200, 10000] # closed interval
    init_method = "bad"
    mode = "total-opt-gap-ratio" # "opt-ratio" or "total-opt-gap-ratio" or "log-opt-gap-ratio"
    file_dirs = ["fig_data_server_0928"] #["fig_data", "fig_data_server_0922", "fig_data_server_0925", "fig_data_server_0928", "fig_data_server_1001"]

    all_batch_means = {}
    for setting_name in settings:
        for policy_name in policies:
            if policy_name in skip_policies:
                continue
            file_prefix = "{}-{}".format(setting_name, policy_name)
            file_paths = []
            for file_dir in file_dirs:
                if note is None:
                    file_paths.extend([os.path.join(file_dir,file_name) for file_name in os.listdir(file_dir)
                                  if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])])
                    # if policy_name in ["id", "lppriority"]:  ### temp, debugging
                    #     file_names = [file_name for file_name in file_names if "testing" in file_name.split("-")]
                else:
                    file_paths.extend([os.path.join(file_dir,file_name) for file_name in os.listdir(file_dir)
                                  if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])
                                  and (note in file_name.split("-")[(-1):])])
            print("{}:{}".format(file_prefix, file_paths))
            if len(file_paths) == 0:
                if policy_name == "whittle":
                    continue
                else:
                    raise FileNotFoundError("no file that match the prefix {} and init_method = {}".format(file_prefix, init_method))
            N2batch_means = {}
            N_longest_T = {} # only plot with the longest T; N_longest_T helps identifying the file with longest T
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    setting_and_data = pickle.load(f)
                    for i, N in enumerate(setting_and_data["Ns"]):
                        if (N < N_range[0]) or (N > N_range[1]):
                            continue
                        if (i, N) not in setting_and_data["full_reward_trace"]:
                            print("N={} not available in {}".format(N, file_path))
                            continue
                        if N not in N2batch_means:
                            N2batch_means[N] = []
                            N_longest_T[N] = 0
                        if N_longest_T[N] > setting_and_data["T"]:
                            continue
                        else:
                            if "save_mean_every" in setting_and_data:
                                save_mean_every = setting_and_data["save_mean_every"]
                            else:
                                save_mean_every = 1
                            if N_longest_T[N] == setting_and_data["T"]:
                                continue #### only use one batch of data with largest horizon; comment out otherwise
                                # print(setting_name, N, "appending data from ", file_path)
                            else:
                                N_longest_T[N] = setting_and_data["T"]
                                N2batch_means[N] = []
                                print(setting_name, N, "replaced with data from ", file_path)
                            if batch_size_mode == "adaptive":
                                batch_size = round(N_longest_T[N] / 20)
                            assert batch_size % save_mean_every == 0, "batch size is not a multiple of save_mean_every={}".format(save_mean_every)
                            for t in range(round(batch_size / save_mean_every)*burn_in_batch, round(setting_and_data["T"] / save_mean_every), round(batch_size / save_mean_every)):
                                N2batch_means[N].append(np.mean(setting_and_data["full_reward_trace"][(i,N)][t:(t+round(batch_size / save_mean_every))]))
            for N in N2batch_means:
                N2batch_means[N] = np.array(N2batch_means[N])
            all_batch_means[(setting_name,policy_name)] = N2batch_means
            with open(file_paths[0], 'rb') as f:
                setting_and_data = pickle.load(f)
                all_batch_means[(setting_name,"upper bound")] = setting_and_data["upper bound"]

    for setting_name in settings:
        # file_prefix = "{}-{}".format(setting_name, "twoset-faithful")
        # file_names = [file_name for file_name in os.listdir(file_dirs[0])
        #               if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])]
        # with open(os.path.join(file_dirs[0], file_names[0]), 'rb') as f:
        #     setting_and_data = pickle.load(f)
        #     upper_bound = setting_and_data["upper bound"]
        upper_bound = all_batch_means[(setting_name,"upper bound")]
        if setting_name in setting2truncate:
            truncate_level = setting2truncate[setting_name]
        else:
            truncate_level = truncate_level_default
        if mode == "opt-ratio":
            plt.plot([N_range[0], N_range[1]], np.array([1, 1]), label="Upper bound", linestyle="--", color="k")
        max_value_for_ylim = 0
        for i, policy_name in enumerate(policies):
            if (policy_name == "whittle") and ((setting_name, policy_name) not in all_batch_means):
                continue
            if policy_name in skip_policies:
                continue
            else:
                Ns_local = []
                avg_rewards_local = []
                yerrs_local = []
                cur_policy_batch_means = all_batch_means[(setting_name, policy_name)]
                for N in cur_policy_batch_means:
                    Ns_local.append(N)
                    avg_rewards_local.append(np.mean(cur_policy_batch_means[N]))
                    yerrs_local.append(1.96 * np.std(cur_policy_batch_means[N]) / np.sqrt(len(cur_policy_batch_means[N])))
                Ns_local = np.array(Ns_local)
                avg_rewards_local = np.array(avg_rewards_local)
                yerrs_local = np.array(yerrs_local)
                sorted_indices = np.argsort(Ns_local)
                Ns_local_sorted = Ns_local[sorted_indices]
                avg_rewards_local_sorted = avg_rewards_local[sorted_indices]
                yerrs_local_sorted = yerrs_local[sorted_indices]
                print(setting_name, policy_name, avg_rewards_local_sorted, yerrs_local_sorted)
                ## special handling of "random-size-8-uniform-(1)":...
                if (setting_name == "random-size-8-uniform-(1)"):
                    if policy_name in ["whittle", "lppriority"]:
                        show_until = bisect.bisect_left(Ns_local_sorted, 6000)
                        Ns_local_sorted = Ns_local_sorted[0:show_until]
                        avg_rewards_local_sorted = avg_rewards_local_sorted[0:show_until]
                        yerrs_local_sorted = yerrs_local_sorted[0:show_until]
                    # for j, N in enumerate(Ns_local_sorted):
                    #     if policy_name in ["whittle", "lppriority"] and (N>=6000):
                            # avg_rewards_local_sorted[j] = upper_bound
                            # yerrs_local_sorted[j] = 0
                        # elif (policy_name == "twoset-faithful") and (N > 8000):
                        #     avg_rewards_local_sorted[j] = upper_bound
                        #     yerrs_local_sorted[j] = 0

                if not plot_CI:
                    if mode == "opt-ratio":
                        cur_curve = plt.plot(Ns_local_sorted, avg_rewards_local_sorted / upper_bound)
                    elif mode == "total-opt-gap-ratio":
                        cur_curve = plt.plot(Ns_local_sorted, (upper_bound - avg_rewards_local_sorted) * Ns_local_sorted / upper_bound)
                        if policy_name not in ["lppriority", "whittle"]:
                            max_value_for_ylim = max(max_value_for_ylim, np.max((upper_bound - avg_rewards_local_sorted) * Ns_local_sorted / upper_bound))
                    elif mode == "log-opt-gap-ratio":
                        cur_curve = plt.plot(Ns_local_sorted, np.log10((upper_bound - avg_rewards_local_sorted)/upper_bound))
                    else:
                        raise NotImplementedError
                    cur_curve.set(label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                marker=policy_markers[i], markersize=8, color=policy_colors[i])
                else:
                    if mode == "opt-ratio":
                        plt.errorbar(Ns_local_sorted, avg_rewards_local_sorted / upper_bound,
                                     yerr=yerrs_local_sorted / upper_bound, label=policy2label[policy_name],
                                     linewidth=1.5, linestyle=linestyle_str[i], marker=policy_markers[i], markersize=8,
                                     color=policy_colors[i])
                    elif mode == "total-opt-gap-ratio":
                        Ns_include_zero = np.insert(Ns_local_sorted, 0, 0)
                        ys = (upper_bound - avg_rewards_local_sorted) * Ns_local_sorted / upper_bound
                        ys_include_zero = np.insert(ys, 0, 0)
                        yerrs_include_zero = np.insert(yerrs_local_sorted * Ns_local_sorted / upper_bound, 0, 0)
                        plt.errorbar(Ns_include_zero, ys_include_zero,
                                     yerr=yerrs_include_zero, label=policy2label[policy_name],
                                     linewidth=1.5, linestyle=linestyle_str[i], marker=policy_markers[i], markersize=8,
                                     color=policy_colors[i])
                        if policy_name not in ["lppriority", "whittle"]:
                            max_value_for_ylim = max(max_value_for_ylim, np.max((upper_bound - avg_rewards_local_sorted+yerrs_local_sorted) * Ns_local_sorted / upper_bound))
                    elif mode == "log-opt-gap-ratio":
                        upper_CI_truncated = np.clip((upper_bound - avg_rewards_local_sorted + yerrs_local_sorted) / upper_bound, truncate_level, None)
                        lower_CI_truncated = np.clip((upper_bound - avg_rewards_local_sorted - yerrs_local_sorted) / upper_bound, truncate_level, None)
                        mean_truncated = np.clip((upper_bound - avg_rewards_local_sorted) / upper_bound, truncate_level, None)
                        print(upper_CI_truncated, lower_CI_truncated)
                        plt.errorbar(Ns_local_sorted, np.log10(mean_truncated),
                                 yerr=np.stack([np.log10(mean_truncated) - np.log10(lower_CI_truncated), np.log10(upper_CI_truncated) - np.log10(mean_truncated)]),
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                    else:
                        raise NotImplementedError
                # print("{} {}, yerrs/upper_bound = {}".format(setting_name, policy_name, yerrs_local_sorted / upper_bound))
                # print("{} {}, relative error = {}".format(setting_name, policy_name, yerrs_local_sorted / np.clip(upper_bound - avg_rewards_local_sorted, 0, None) / np.log(10)))
            plt.xlabel("N", fontsize=14)
        plt.xticks(fontsize=14)
        if mode == "opt-ratio":
            plt.ylabel("Optimality ratio", fontsize=14)
        elif mode == "total-opt-gap-ratio":
            plt.ylabel("Total optimality gap ratio", fontsize=14)
        elif mode == "log-opt-gap-ratio":
            plt.ylabel("Log optimality gap ratio", fontsize=14)
        else:
            raise NotImplementedError
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.grid()
        if (setting_name == "random-size-8-uniform-(1)") and (mode == "log-opt-gap-ratio"):
            pass # plt.legend(fontsize=14, loc="center right")
        else:
            plt.legend(fontsize=14)
        if mode == "opt-ratio":
            # plt.savefig("figs3/{}-N{}-{}-{}.pdf".format(setting_name, Ns[0], Ns[-1], init_method))
            plt.savefig("figs3/{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
            plt.savefig("formal_figs_exponential/{}.png".format(setting_name))
        elif mode == "total-opt-gap-ratio":
            plt.ylim((-0.06*max_value_for_ylim, max_value_for_ylim*1.05))
            plt.savefig("figs3/total-gap-ratio-{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
            plt.savefig("formal_figs_exponential/total-gap-ratio-{}.png".format(setting_name))
        elif mode == "log-opt-gap-ratio":
            plt.ylim((np.log10(truncate_level)+0.1, None))
            plt.savefig("figs3/log-gap-ratio-{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
            plt.savefig("formal_figs_exponential/log-gap-ratio-{}.png".format(setting_name))
        else:
            raise NotImplementedError
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=600)
    # for setting_name in ["non-sa"]: #["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
    #     for policy_name in ["ftva"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "whittle"]: # ["id", "setexp", "setopt", "ftva", "lppriority", "setexp-priority", "twoset-v1"]
    #         for rep_id in range(3,6):
    #             run_policies(setting_name, policy_name, "random", 160000, note="T16e4r{}".format(rep_id))

    ## random 10-state dirichlet examples
    # for i in [276]: #[137, 355, 582]:
    #     setting_name = "random-size-10-dirichlet-0.05-({})".format(i)
    #     setting_path = "setting_data/local_unstable_subopt/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["lppriority"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "whittle"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         for rep_id in range(1,6):
    #             run_policies(setting_name, policy_name, "random", 20000, setting_path, note="T2e4r{}".format(rep_id))

    # for setting_name in ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
    #     for policy_name in ["id", "setexp", "setopt", "ftva", "lppriority", "setexp-priority", "twoset-v1"]:
    #         for rep_id in range(1,6):
    #             run_policies(setting_name, policy_name, "random", 20000, note="T2e4r{}".format(rep_id))

    # ## random three-state examples
    # for i in range(5):
    #     setting_name = "random-size-3-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     # print(setting.suggest_act_frac, (1000*setting.suggest_act_frac).is_integer())
    #     # setting.suggest_act_frac = int(100*setting.suggest_act_frac)/100
    #     # setting.distr = "uniform"
    #     # rb_settings.save_bandit(setting, setting_path, None)
    #     for policy_name in ["twoset-faithful", "id", "lppriority", "whittle", "ftva"]:
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(100, 1100, 100)), note="T4e4")
    #         if i==2:
    #             run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(2000, 20000, 1000)), note="T4e4")

    # for setting_name in ["new2-eight-states-045"]:
    #     for policy_name in ["twoset-faithful"]: #["lppriority", "twoset-v1", "twoset-faithful", "id", "setexp", "ftva", "setexp-priority"]:
    #         run_policies(setting_name, policy_name, "bad", 40000, note="T4e4", Ns=list(range(100,4000,200)), debug=True)

    # for setting_name in ["conveyor-belt-nd-12"]:
    #     for policy_name in ["twoset-faithful"]: #["lppriority", "twoset-v1", "twoset-faithful", "id", "setexp", "ftva", "setexp-priority"]:
    #         run_policies(setting_name, policy_name, "bad", 40000, note="T4e4", Ns=list(range(100,4000,200)), debug=True)

    # Ns = list(range(200,1200,200)) + [1500] + list(range(2000, 12000,2000))
    # for i in range(2,3):
    #     setting_name = "random-size-8-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["lppriority"]: #, "id", "lppriority", "whittle", "ftva"]:
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=Ns, note="testing")


    my_pool = mp.Pool(6)

    task_list = []
    Ns = [10000]
    for i in range(0, 35):
        setting_name = "random-size-10-dirichlet-0.05-({})".format(i)
        setting_path = "setting_data/unselected/" + setting_name
        if os.path.exists(setting_path):
            setting = rb_settings.ExampleFromFile(setting_path)
        else:
            print("{} not found!!".format(setting_path))
            continue
        for policy_name in ["twoset-faithful", "ftva", "id", "lppriority", "whittle"]:
            T = 80000
            note = "T8e4"
            cur_save_path = "{}/{}-{}-N{}-{}-{}-{}".format("fig_data_250327", setting_name, policy_name, Ns[0], Ns[-1], "random", note)
            if (not os.path.exists(cur_save_path)) and (not (policy_name == "whittle")):
                print(cur_save_path, "is not simulated yet")
                # run_policies(setting_name, policy_name, "random", T, setting_path=setting_path, Ns=Ns, save_dir="fig_data_250327", no_run=False, note=note)
                # print()
                task_list.append(
                    my_pool.apply_async(run_policies, args=(setting_name, policy_name, "random", T, setting_path, Ns),
                                        kwds={"save_dir": "fig_data_250327", "note": note})
                )

    my_pool.close()
    my_pool.join()

