import numpy as np
import cvxpy as cp
import scipy

import understand_assumptions
from discrete_RB import *
import rb_settings
import time
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import os


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, Ns=None, skip_N_below=None, no_run=False, debug=False, note=None):
    if setting_name == "eight-states":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    elif setting_name == "three-states":
        setting = rb_settings.Gast20Example2()
    elif setting_name == "non-sa":
        setting = rb_settings.NonSAExample()
    elif setting_name == "eight-states-045":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.suggest_act_frac = 0.45
    elif setting_name == "new-eight-states":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.02
    elif setting_name == "new2-eight-states":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.1/30
    elif setting_name == "new2-eight-states-045":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
        setting.reward_tensor[0,1] = 0.1/30
        setting.suggest_act_frac = 0.45
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
    print()
    rb_settings.print_bandit(setting)

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    print("min(y(sneu,1), y(sneu,0)) = ", understand_assumptions.y2nondegeneracy(y))
    W = analyzer.compute_W(abstol=1e-10)[0]
    # print("W=", W)
    print("2*lambda_W = ", 2*np.linalg.norm(W, ord=2))
    if setting_name == "eight-states":
        priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
    else:
        priority_list = analyzer.solve_LP_Priority()
    print("priority list =", priority_list)
    whittle_priority = analyzer.solve_whittles_policy()
    print("Whittle priority=", whittle_priority)
    U = analyzer.compute_U(abstol=1e-10)[0]
    # print("U=\n", U)
    if U is not np.infty:
        print("spectral norm of U=", np.max(np.abs(np.linalg.eigvals(U))))
    else:
        print("U diverges, locally unstable")

    if no_run:
        return

    if note is None:
        data_file_name = "fig_data/{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
    else:
        data_file_name = "fig_data/{}-{}-N{}-{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
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
                init_states = np.random.choice(np.arange(4, 8), N, replace=True)
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
                for t in range(T):
                    cur_states = rb.get_states()
                    actions, num_ideal_acts = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(num_ideal_acts)
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
            elif policy_name == "setexp":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
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
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
                    #     print("focus set size before rounding={}, after rounding={}".format(policy.m.value*N, len(focus_set)))
                    #     print("conformity count = {}, non-shrink count = {}".format(conformity_count, non_shrink_count))
            elif policy_name == "setopt":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
                    # if t%100 == 0:
                    #     sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
                    #     s_fracs = np.sum(sa_fracs, axis=1)
                    #     print("t={}\ns_fracs={}".format(t, s_fracs))
                    #     print("focus set size before rounding={}, after rounding={}".format(policy.m.value*N, len(focus_set)))
                    #     print("conformity count = {}".format(conformity_count))
            elif policy_name == "setopt-tight":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states, subproblem="tight") ##
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
            elif policy_name == "setexp-id":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
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
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
            elif policy_name == "setopt-id":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="ID")
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
            elif policy_name == "setexp-priority":
                policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
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
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
            elif policy_name == "setopt-priority":
                policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
                for t in range(T):
                    cur_states = rb.get_states()
                    focus_set = policy.get_new_focus_set(cur_states=cur_states)
                    actions, conformity_flag = policy.get_actions(cur_states, focus_set, tb_rule="priority",
                                                                  tb_priority=priority_list)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(focus_set))
            elif policy_name == "ftva":
                policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N,
                                    act_frac=act_frac, init_virtual=None)
                for t in range(T):
                    prev_state = rb.get_states()
                    actions, virtual_actions = policy.get_actions(prev_state)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    num_good_arms = np.sum(np.all([actions==virtual_actions, prev_state==policy.virtual_states], axis=0))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    new_state = rb.get_states()
                    policy.virtual_step(prev_state, new_state, actions, virtual_actions)
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(num_good_arms)
            elif policy_name == "lppriority":
                policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
            elif policy_name == "whittle":
                if whittle_priority is -1:
                    print("Non-indexable!!!")
                    return
                elif whittle_priority is -2:
                    print("Multichain!!!")
                    return
                else:
                    policy = PriorityPolicy(setting.sspa_size, whittle_priority, N=N, act_frac=act_frac)
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
            elif policy_name == "twoset-v1":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
                # print("eta=", policy.eta)
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(OL_set))
            elif policy_name == "twoset-integer":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U, rounding="misocp")
                # print("eta=", policy.eta)
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(OL_set))
            elif policy_name == "twoset-faithful":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
                # print("eta=", policy.eta)
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set)
                    actions, EP_set = policy.get_actions_with_EP_set(cur_states=cur_states, cur_OL_set=OL_set, last_EP_set=EP_set)
                    assert np.sum(actions) == round(act_frac*N), "Global budget inconsistent: {}!={}".format(np.sum(actions), round(act_frac*N))
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(OL_set))
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
                        "upper bound": analyzer.opt_value
                    }
                with open(data_file_name, 'wb') as f:
                    pickle.dump(setting_and_data, f)


def figure_from_multiple_files():
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["random-size-3-uniform-({})".format(i) for i in range(5)]
    settings = ["random-size-3-uniform-({})".format(i) for i in range(5)]
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id"]
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


def figure_from_multiple_files_no_CI():
    """
    A simpler plotting function without computing confidence interval
    """
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["random-size-3-uniform-({})".format(i) for i in range(5)]
    settings = ["random-size-3-uniform-({})".format(i) for i in [2]] #["stable-size-10-dirichlet-0.05-({})".format(i) for i in [4116]] # [4116, 2667] [2270, 9632]   #["random-size-3-uniform-({})".format(i) for i in range(0,5)] #["new2-eight-states-045"] #
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id", "twoset-v1", "twoset-faithful"]
    skip_policies =  ["ftva", "setexp", "setexp-priority","twoset-v1"] #, "lppriority", "whittle", "id"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-", "-", "-."]
    policy_markers = ["v",".","^","s","p","*", "v", "P"]
    policy_colors = ["m","c","y","r","g","b", "brown", "orange"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)",
                    "twoset-v1":"Two-set v1", "twoset-faithful":"Two-set policy"}
    target_num_batches = 20
    Ns = np.array(list(range(2000, 20000,1000))) #np.array(list(range(100,1100,100))) #np.array(list(range(1500, 5500, 500))) # list(range(1000, 20000, 1000))
    init_method = "random"
    mode = "total-opt-gap" # "opt-ratio" or "total-opt-gap" or "log-opt-gap"

    avg_rewards = {}
    for setting_name in settings:
        for policy_name in policies:
            if policy_name in skip_policies:
                continue
            file_prefix = "{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
            file_dir = "fig_data"
            file_names = [file_name for file_name in os.listdir(file_dir) if file_name.startswith(file_prefix)]
            print("{}:{}".format(file_prefix, file_names))
            if len(file_names) == 0:
                if policy_name == "whittle":
                    continue
                else:
                    raise FileNotFoundError("no file with prefix {}".format(file_prefix))
            total_avg_rewards = np.zeros(len(Ns))
            for file_name in file_names:
                with open(os.path.join(file_dir, file_name), 'rb') as f:
                    setting_and_data = pickle.load(f)
                    total_avg_rewards += np.average(setting_and_data["reward_array"], axis=0)
            avg_rewards[(setting_name,policy_name)] = total_avg_rewards / len(file_names)

    for setting_name in settings:
        if setting_name == "eight-states":
            upper_bound = 0.0125
        elif setting_name == "three-states":
            upper_bound = 0.12380016733626052
        elif setting_name == "non-sa":
            upper_bound = 1
        else:
            files_w_prefix = [filename for filename in os.listdir("fig_data")
                              if filename.startswith("{}-{}-N{}-{}-{}".format(setting_name, "twoset-faithful", Ns[0], Ns[-1], init_method))]
            with open("fig_data/"+files_w_prefix[0], 'rb') as f:
                setting_and_data = pickle.load(f)
                upper_bound = setting_and_data["upper bound"]
        if mode == "opt-ratio":
            plt.plot(Ns, np.array([1] * len(Ns)), label="Upper bound", linestyle="--", color="k")
        for i,policy_name in enumerate(policies):
            if (policy_name == "whittle") and ((setting_name, policy_name) not in avg_rewards):
                continue
            if policy_name in skip_policies:
                continue
            else:
                if mode == "opt-ratio":
                    plt.plot(Ns, avg_rewards[(setting_name,policy_name)] / upper_bound,
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                elif mode == "total-opt-gap":
                    plt.plot(Ns, (upper_bound - avg_rewards[(setting_name,policy_name)]) * Ns,
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                elif mode == "log-opt-gap":
                    plt.plot(Ns, np.log10(upper_bound - avg_rewards[(setting_name,policy_name)]),
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                else:
                    raise NotImplementedError
            plt.xlabel("N", fontsize=14)
        plt.xticks(fontsize=14)
        if mode == "opt-ratio":
            plt.ylabel("Optimality ratio", fontsize=14)
        elif mode == "total-opt-gap":
            plt.ylabel("N * optimality gap ")
        elif mode == "log-opt-gap":
            plt.ylabel("Log_10 optimality gap")
        else:
            raise NotImplementedError
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.grid()
        plt.legend(fontsize=14)
        if mode == "opt-ratio":
            # plt.savefig("figs3/{}-N{}-{}-{}.pdf".format(setting_name, Ns[0], Ns[-1], init_method))
            plt.savefig("figs3/{}-N{}-{}-{}.png".format(setting_name, Ns[0], Ns[-1], init_method))
        elif mode == "total-opt-gap":
            plt.savefig("figs3/total-gap-{}-N{}-{}-{}.png".format(setting_name, Ns[0], Ns[-1], init_method))
        elif mode == "log-opt-gap":
            plt.savefig("figs3/log-gap-{}-N{}-{}-{}.png".format(setting_name, Ns[0], Ns[-1], init_method))
        else:
            raise NotImplementedError
        plt.show()


def figure_from_multiple_files_flexible_N_no_CI():
    """
    Plotting function that reads data files with Ns to combine into one plot.
    """
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["random-size-3-uniform-({})".format(i) for i in range(5)]
    settings = ["random-size-8-uniform-({})".format(i) for i in range(0,4)]  # ["mix-random-size-10-dirichlet-0.05-({})-(2270)-ratio-0.95".format(i) for i in [1436, 6265]] #["stable-size-10-dirichlet-0.05-({})".format(i) for i in [4339]]#, 4149, 4116, 2667, 2270, 9632]]   #["random-size-3-uniform-({})".format(i) for i in range(0,5)] #["new2-eight-states-045"]
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id", "twoset-v1", "twoset-faithful"]
    skip_policies =  ["setexp", "setexp-priority","twoset-v1"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-", "-", "-."]
    policy_markers = ["v",".","^","s","p","*", "v", "P"]
    policy_colors = ["m","c","y","r","g","b", "brown", "orange"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)",
                    "twoset-v1":"Two-set v1", "twoset-faithful":"Two-set policy"}
    target_num_batches = 20
    mode = "fixed" # "fixed", or "range"
    N_range = [200, 10000] # closed interval
    # Ns = list(range(2000,20000,1000))
    init_method = "random"
    mode = "log-opt-gap" # "opt-ratio" or "total-opt-gap" or "log-opt-gap"

    avg_rewards = {}
    for setting_name in settings:
        for policy_name in policies:
            if policy_name in skip_policies:
                continue
            file_prefix = "{}-{}".format(setting_name, policy_name)
            file_dir = "fig_data"
            file_names = [file_name for file_name in os.listdir(file_dir)
                          if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])]
            print("{}:{}".format(file_prefix, file_names))
            if len(file_names) == 0:
                if policy_name == "whittle":
                    continue
                else:
                    raise FileNotFoundError("no file that match the prefix {} and init_method = {}".format(file_prefix, init_method))
            N2reward = {}
            for file_name in file_names:
                with open(os.path.join(file_dir, file_name), 'rb') as f:
                    setting_and_data = pickle.load(f)
                    for i, N in enumerate(setting_and_data["Ns"]):
                        if (N < N_range[0]) or (N > N_range[1]):
                            continue
                        if N in N2reward:
                            N2reward[N]["rewards"].extend(setting_and_data["reward_array"][:,i])
                            N2reward[N]["Ts"].extend([setting_and_data["T"]]*num_reps)
                        else:
                            N2reward[N] = {"Ts":[], "rewards":[], "average":0}  # T, and avg reward
                            num_reps = len(setting_and_data["reward_array"][:,i])
                            N2reward[N]["rewards"].extend(setting_and_data["reward_array"][:,i])
                            N2reward[N]["Ts"].extend([setting_and_data["T"]]*num_reps)
            for N in N2reward:
                N2reward[N]["average"] = np.sum(np.array(N2reward[N]["Ts"]) * np.array(N2reward[N]["rewards"])) / np.sum(np.array(N2reward[N]["Ts"]))
            avg_rewards[(setting_name,policy_name)] = N2reward

    for setting_name in settings:
        if setting_name == "eight-states":
            upper_bound = 0.0125
        elif setting_name == "three-states":
            upper_bound = 0.12380016733626052
        elif setting_name == "non-sa":
            upper_bound = 1
        else:
            file_prefix = "{}-{}".format(setting_name, "twoset-faithful")
            file_dir = "fig_data"
            file_names = [file_name for file_name in os.listdir(file_dir)
                          if file_name.startswith(file_prefix) and (init_method in file_name.split("-")[(-2):])]
            with open(os.path.join("fig_data", file_names[0]), 'rb') as f:
                setting_and_data = pickle.load(f)
                upper_bound = setting_and_data["upper bound"]
        if mode == "opt-ratio":
            plt.plot([N_range[0], N_range[1]], np.array([1, 1]), label="Upper bound", linestyle="--", color="k")
        for i, policy_name in enumerate(policies):
            if (policy_name == "whittle") and ((setting_name, policy_name) not in avg_rewards):
                continue
            if policy_name in skip_policies:
                continue
            else:
                Ns_local = []
                avg_rewards_local = []
                for N in avg_rewards[(setting_name, policy_name)]:
                    Ns_local.append(N)
                    avg_rewards_local.append(avg_rewards[(setting_name, policy_name)][N]["average"])
                Ns_local = np.array(Ns_local)
                avg_rewards_local = np.array(avg_rewards_local)
                sorted_indices = np.argsort(Ns_local)
                Ns_local_sorted = Ns_local[sorted_indices]
                avg_rewards_local_sorted = avg_rewards_local[sorted_indices]
                print(Ns_local_sorted.shape)
                print(avg_rewards_local_sorted.shape)

                if mode == "opt-ratio":
                    plt.plot(Ns_local_sorted, avg_rewards_local_sorted / upper_bound,
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                elif mode == "total-opt-gap":
                    plt.plot(Ns_local_sorted, (upper_bound - avg_rewards_local_sorted) * Ns_local_sorted,
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                elif mode == "log-opt-gap":
                    plt.plot(Ns_local_sorted, np.log10(upper_bound - avg_rewards_local_sorted),
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                else:
                    raise NotImplementedError
            plt.xlabel("N", fontsize=14)
        plt.xticks(fontsize=14)
        if mode == "opt-ratio":
            plt.ylabel("Optimality ratio", fontsize=14)
        elif mode == "total-opt-gap":
            plt.ylabel("N * optimality gap ")
        elif mode == "log-opt-gap":
            plt.ylabel("Log_10 optimality gap")
        else:
            raise NotImplementedError
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.grid()
        plt.legend(fontsize=14)
        if mode == "opt-ratio":
            # plt.savefig("figs3/{}-N{}-{}-{}.pdf".format(setting_name, Ns[0], Ns[-1], init_method))
            plt.savefig("figs3/{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
        elif mode == "total-opt-gap":
            plt.savefig("figs3/total-gap-{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
        elif mode == "log-opt-gap":
            plt.savefig("figs3/log-gap-{}-N{}-{}-{}.png".format(setting_name, N_range[0], N_range[1], init_method))
        else:
            raise NotImplementedError
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=600)
    # for setting_name in ["non-sa"]: #["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
    #     for policy_name in ["ftva"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "whittle"]: # ["id", "setexp", "setopt", "ftva", "lppriority", "setexp-priority", "twoset-v1"]
    #         for rep_id in range(3,6):
    #             tic = time.time()
    #             run_policies(setting_name, policy_name, "random", 160000, note="T16e4r{}".format(rep_id))
    #             toc = time.time()
    #             print("when T=160000, total time per policy =", toc-tic)

    # ## random 10-state dirichlet examples
    # for i in [582]: #[137, 355, 582]:
    #     setting_name = "random-size-10-dirichlet-0.05-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["setexp-priority"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "whittle"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         for rep_id in range(1,6):
    #             run_policies(setting_name, policy_name, "random", 20000, setting_path, note="T2e4r{}".format(rep_id))

    # for setting_name in ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
    #     for policy_name in ["id", "setexp", "setopt", "ftva", "lppriority", "setexp-priority", "twoset-v1"]:
    #         for rep_id in range(1,6):
    #             tic = time.time()
    #             run_policies(setting_name, policy_name, "random", 20000, note="T2e4r{}".format(rep_id))
    #             toc = time.time()
    #             print("when T=20000, total time per policy =", toc-tic)

    # ## random three-state examples
    # for i in range(5):
    #     setting_name = "random-size-3-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     # print(setting.suggest_act_frac, (1000*setting.suggest_act_frac).is_integer())
    #     # setting.suggest_act_frac = int(100*setting.suggest_act_frac)/100
    #     # setting.distr = "uniform"
    #     # rb_settings.save_bandit(setting, setting_path, None)
    #     for policy_name in ["twoset-faithful"]: #["id", "lppriority", "twoset-v1", "twoset-faithful", "whittle"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "twoset-v1", "twoset-faithful", "whittle"]:
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(100, 1100, 100)), note="T4e4")
    #         if i==2:
    #             run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(2000, 20000, 1000)), note="T4e4")

    # for setting_name in ["new2-eight-states-045"]:
    #     for policy_name in ["twoset-faithful"]: #["lppriority", "twoset-v1", "twoset-faithful", "id", "setexp", "ftva", "setexp-priority"]:
    #         tic = time.time()
    #         run_policies(setting_name, policy_name, "bad", 40000, note="T4e4", Ns=list(range(100,4000,200)))
    #         toc = time.time()
    #         print("when T=40000, total time per policy =", toc-tic)

    # ## random 10-state dirichlet examples
    ## [4116, 2667], rho(Phi) ~= 0.90, Ns = list(range(1000, 21000,1000))
    ## [2270, 9632], rho(Phi) ~= 0.95, Ns = list(range(1000, 42000,2000))
    ## [4339, 4149], rho(Phi) ~= 0.99, Ns = list(range(1000, 84000,4000))
    # for i in [4339, 4149, 2667, 2270, 9632]:
    #     setting_name = "stable-size-10-dirichlet-0.05-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["twoset-faithful", "lppriority"]: #["twoset-faithful", "id", "lppriority", "whittle"]:
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(100, 1100, 100)), skip_N_below=None, no_run=False)

    # for i in [1436, 6265]:
    #     setting_name = "mix-random-size-10-dirichlet-0.05-({})-(2270)-ratio-0.95".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["twoset-faithful"]: #["twoset-faithful", "id", "lppriority", "whittle"]:
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=list(range(100, 1100, 100)), skip_N_below=300, no_run=False)

    # Ns = list(range(200,1200,200)) + [1500] + list(range(2000, 12000,2000))
    # for i in range(1,2):
    #     setting_name = "random-size-8-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["twoset-faithful"]:#, "id", "lppriority", "whittle", "ftva"]: #["id", "setexp", "ftva", "lppriority", "setexp-priority", "twoset-v1", "twoset-faithful", "whittle"]:
    #         tic = time.time()
    #         run_policies(setting_name, policy_name, "random", 40000, setting_path, Ns=Ns, note="T4e4", no_run=False)
    #         print("time for running one policy with T=40000 and {} data points is {}".format(len(Ns), time.time()-tic))


    figure_from_multiple_files_flexible_N_no_CI()
