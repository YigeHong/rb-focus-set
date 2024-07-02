import numpy as np
import cvxpy as cp
import scipy
import pickle
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
# import ffmpeg
from discrete_RB import *
import rb_settings
# from find_more_counterexamples import test_local_stability

def test_repeated_solver():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    N = 100
    act_frac = setting.suggest_act_frac

    sspa = list(range(setting.sspa_size))
    # init_states = np.random.choice(sspa, N, replace=True)
    # rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    W = analyzer.compute_W(abstol=1e-10)[0]
    setexp_policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)
    setopt_policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)

    cum_time = 0
    T = 10
    for i in range(T):
        tic = time.time()
        states = np.random.choice(sspa, N, replace=True)
        setexp_policy.get_new_focus_set(states=states, last_focus_set=np.array([], dtype=int))
        setopt_policy.get_new_focus_set(states=states)
        print("difference between two policies Xt(Dt): ", np.linalg.norm(setexp_policy.z.value - setopt_policy.z.value, ord=2))
        print()
        itime = time.time() - tic
        print("{}-th solving time = ".format(i), itime)
        cum_time += itime
    print("total time = ", cum_time)
    print("time per iter = ", cum_time / T)


def test_W_solver():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting = rb_settings.Gast20Example2()
    N = 100
    act_frac = setting.suggest_act_frac

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    analyzer.solve_lp()

    W, spn_error = analyzer.compute_W(abstol=1e-10)
    print("W={}, lamw = {}, spectral norm error<={}".format(W, np.linalg.norm(W, ord=2), spn_error))

    W_sqrt = scipy.linalg.sqrtm(W)
    print("W_sqrt = ", W_sqrt)
    print(np.sort(np.linalg.eigvals(W)))
    print(np.sort(np.linalg.eigvals(W_sqrt)))


def test_compute_future_max_req():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting = rb_settings.Gast20Example2()
    N = 100
    act_frac = setting.suggest_act_frac
    T_ahead = 100
    num_points_show = 1
    initialization = "transient" # or "transient" or "steady-state"

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    analyzer.solve_lp()

    Ts = np.arange(0, T_ahead)
    budget_line = np.array([act_frac] * T_ahead)
    plt.plot(Ts, budget_line, linestyle="--", label="budget")

    for i in range(num_points_show):
        if initialization == "transient":
            init_state_fracs = np.random.uniform(0, 1, size=(setting.sspa_size,))
            init_state_fracs = init_state_fracs / np.sum(init_state_fracs)
        elif initialization == "steady-state":
            init_state_fracs = np.random.multinomial(N, analyzer.state_probs) / N
        else:
            raise NotImplementedError
        future_reqs =  analyzer.get_future_expected_budget_requirements(init_state_fracs, T_ahead)
        print("initial_frequenty=", init_state_fracs)
        plt.plot(Ts, future_reqs, label="requirement")

        Lone_upper, Lone_lower = analyzer.get_future_budget_req_bounds_Lone(init_state_fracs, T_ahead)
        plt.plot(Ts, Lone_upper, linestyle=":", label="Lone upper")
        plt.plot(Ts, Lone_lower, linestyle=":", label="Lone lower")

        Wnorm_upper, Wnorm_lower = analyzer.get_future_budget_req_bounds_Wnorm(init_state_fracs, T_ahead)
        plt.plot(Ts, Wnorm_upper, linestyle="-.", label="W norm upper")
        plt.plot(Ts, Wnorm_lower, linestyle="-.", label="W norm lower")

    plt.legend()
    plt.show()


# def test_UGAP():
#     # probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
#     # setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
#     # setting = rb_settings.Gast20Example2()
#     setting = rb_settings.NonSAExample()
#     N = 100
#     act_frac = setting.suggest_act_frac
#
#     # init_states = np.random.choice(sspa, N, replace=True)
#     # rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)
#
#     analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
#     y = analyzer.solve_lp()[1]
#     W = analyzer.compute_W(abstol=1e-10)[0]
#     print("W=", W)
#     priority = analyzer.solve_LP_Priority()
#     print("LP Priority=", priority)
#
#     init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
#
#     test_local_stability(setting, priority, act_frac, try_steps=2000, eps=1e-3, )

def edit_data():
    with open("fig_data/100-300-random-size-3-uniform-(1)-twoset-v1-N100-1000-random", "rb") as f:
        data_dict_1 = pickle.load(f)
        print(data_dict_1)
    with open("fig_data/400-500-random-size-3-uniform-(1)-twoset-v1-N100-1000-random", "rb") as f:
        data_dict_2 = pickle.load(f)
    with open("fig_data/600-800-(2000)-random-size-3-uniform-(1)-twoset-v1-N100-1000-random", "rb") as f:
        data_dict_3 = pickle.load(f)
    with open("fig_data/900-1000-(2000)-random-size-3-uniform-(1)-twoset-v1-N100-1000-random", "rb") as f:
        data_dict_4 = pickle.load(f)
    data_dict = data_dict_4.copy()
    data_dict["reward_array"][0,0:3] = data_dict_1["reward_array"][0,0:3]
    data_dict["reward_array"][0,3:5] = data_dict_2["reward_array"][0,3:5]
    data_dict["reward_array"][0,5:8] = data_dict_3["reward_array"][0,5:8]
    data_dict["reward_array"][0,8:10] = data_dict_4["reward_array"][0,8:10]
    with open("fig_data/random-size-3-uniform-(1)-twoset-v1-N100-1000-random", "wb") as f:
        pickle.dump(data_dict, f)

def test_run_policies():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting.suggest_act_frac = 0.45
    # setting = rb_settings.Gast20Example2()
    # setting = rb_settings.NonSAExample()
    # setting = rb_settings.ExampleFromFile("setting_data/random-size-3-uniform-(0)")
    N = 5000
    act_frac = setting.suggest_act_frac
    rb_settings.print_bandit(setting)

    # init_states = np.random.choice(sspa, N, replace=True)
    # rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("W=", W)
    if (type(setting) == rb_settings.ConveyorExample) and (setting.suggest_act_frac == 0.5):
        priority = analyzer.solve_LP_Priority(fixed_dual=0)
    else:
        priority = analyzer.solve_LP_Priority()
    print("LP Priority=", priority)
    whittle_priority = analyzer.solve_whittles_policy()
    print("Whittle priority=", whittle_priority)
    U = analyzer.compute_U(abstol=1e-10)[0]
    print("U=\n", U)

    # init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
    init_states = np.random.choice(np.arange(4, setting.sspa_size), N, replace=True)
    # init_states = 4*np.ones((N,))
    # init_states[0:int(N/3)] = 5
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)

    id_policy = IDPolicy(setting.sspa_size, y, N, act_frac)
    setexp_policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac, W=W)
    setopt_policy = SetOptPolicy(setting.sspa_size, y, N, act_frac, W)
    twoset_policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
    print("eta=", twoset_policy.eta)

    T = 2000
    total_reward = 0
    conformity_count = 0
    non_shrink_count = 0
    focus_set =np.array([], dtype=int)
    OL_set =np.array([], dtype=int)
    total_focus_set_size = 0
    total_OL_set_size = 0
    priority_policy = PriorityPolicy(setting.sspa_size, priority, N=N, act_frac=act_frac)
    # for t in range(T):
    #     cur_states = rb.get_states()
    #     actions = priority_policy.get_actions(cur_states)
    #     instant_reward = rb.step(actions)
    #     total_reward += instant_reward
    #     if t%100 == 0:
    #         sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
    #         s_fracs = np.sum(sa_fracs, axis=1)
    #         print("t={}\ns_fracs={}".format(t, s_fracs))
    # print("avg reward = {}".format(total_reward / T))
    # total_reward = 0
    # for t in range(T):
    #     cur_states = rb.get_states()
    #     actions = id_policy.get_actions(cur_states)
    #     instant_reward = rb.step(actions)
    #     total_reward += instant_reward
    #     if t%100 == 0:
    #         sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
    #         s_fracs = np.sum(sa_fracs, axis=1)
    #         print("t={}\ns_fracs={}".format(t, s_fracs))
    # print("avg reward = {}".format(total_reward / T))
    # total_reward = 0
    # for t in range(T):
    #     cur_states = rb.get_states()
    #     focus_set, non_shrink_flag = setexp_policy.get_new_focus_set(cur_states=cur_states, last_focus_set=focus_set)
    #     actions, conformity_flag = setexp_policy.get_actions(cur_states, focus_set, tb_rule="priority", tb_priority=priority)
    #     conformity_count += conformity_flag
    #     non_shrink_count += non_shrink_flag
    #     instant_reward = rb.step(actions)
    #     total_reward += instant_reward
    #     total_focus_set_size += len(focus_set)
    #     if t%100 == 0:
    #         sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
    #         s_fracs = np.sum(sa_fracs, axis=1)
    #         print("t={}\ns_fracs={}".format(t, s_fracs))
    #         print("focus set size before rounding={}, after rounding={}".format(setexp_policy.m.value*N, len(focus_set)))
    #         print("conformity count = {}, non-shrink count = {}".format(conformity_count, non_shrink_count))
            ### the next block of code plots the future expected budget requirement vs the L1 norm bound starting from t
            # T_ahead = 20
            # Ts = np.arange(0, T_ahead)
            # budget_line = np.array([act_frac] * T_ahead)
            # plt.plot(Ts, budget_line, linestyle="--", label="budget")
            # future_reqs =  analyzer.get_future_expected_budget_requirements(s_fracs, T_ahead)
            # plt.plot(Ts, future_reqs, label="requirement")
            # Lone_upper, Lone_lower = analyzer.get_future_budget_req_bounds_Lone(s_fracs, T_ahead)
            # plt.plot(Ts, Lone_upper, linestyle=":", label="Lone upper")
            # plt.plot(Ts, Lone_lower, linestyle=":", label="Lone lower")
            # plt.legend()
            # plt.show()
    # print("avg reward = {}".format(total_reward / T))
    # print("avg focus set size = {}".format(total_focus_set_size / T))
    # print("non-shrinking count = {}".format(non_shrink_count))
    # total_reward = 0
    # for t in range(T):
    #     cur_states = rb.get_states()
    #     focus_set = setopt_policy.get_new_focus_set(cur_states=cur_states) ###
    #     actions, conformity_flag = setopt_policy.get_actions(cur_states, focus_set, tb_rule="priority", tb_priority=priority)
    #     # focus_set, focus_set_outer = setopt_policy.get_new_focus_set_two_stage(cur_states=cur_states) ###
    #     # actions, conformity_flag = setopt_policy.get_actions_two_stage(cur_states, focus_set, focus_set_outer)
    #     conformity_count += conformity_flag
    #     instant_reward = rb.step(actions)
    #     total_reward += instant_reward
    #     total_focus_set_size += len(focus_set)
    #     if t%100 == 0:
    #         sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
    #         s_fracs = np.sum(sa_fracs, axis=1)
    #         print("t={}\ns_fracs={}".format(t, s_fracs))
    #         print("focus set size before rounding={}, after rounding={}".format(setopt_policy.m.value*N, len(focus_set)))
    #         print("conformity count = {}".format(conformity_count))
    # print("avg reward = {}".format(total_reward / T))
    # print("avg focus set size = {}".format(total_focus_set_size / T))
    # total_reward = 0
    # for t in range(T):
    #     cur_states = rb.get_states()
    #     OL_set = twoset_policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
    #     actions = twoset_policy.get_actions(cur_states, OL_set)
    #     instant_reward = rb.step(actions)
    #     total_reward += instant_reward
    #     total_OL_set_size += len(OL_set)
    #     if t%100 == 0:
    #         sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
    #         s_fracs = np.sum(sa_fracs, axis=1)
    #         print("t={}\ns_fracs={}".format(t, s_fracs))
    #         print("OL set size before rounding={}, after rounding={}".format((twoset_policy.m.value or 0)*N, len(OL_set)))
    # print("avg reward = {}".format(total_reward / T))
    # print("avg OL set size = {}".format(total_OL_set_size / T))


def get_ID_max_norm_focus_set(cur_states, beta, opt_state_probs, norm, W=None, ratiocw=None):
    """
    given the current states cur_states, calculate the function n-> max_{n'<=n}||cur_states[0:n'] - mu*n'/N||_1 / 2, and find the
    largest n such that the function is below the curve n->beta*(1-n/N)
    """
    sspa_size = len(opt_state_probs)
    N = len(cur_states)
    norm_bounds = []
    for n in range(1,N+1):
        XD = states_to_scaled_state_counts(sspa_size, N, cur_states[0:n])
        if norm == "L1":
            norm_bounds.append(np.linalg.norm(XD - n/N*opt_state_probs, ord=1) / 2)
        elif norm == "W":
            xmmu_diff = XD - n/N*opt_state_probs
            norm_bounds.append(np.sqrt(np.matmul(np.matmul(xmmu_diff.T, W), xmmu_diff))/2 * ratiocw)
        else:
            raise NotImplementedError
    norm_bounds = np.array(norm_bounds)
    max_norm_bounds = np.array([np.max(norm_bounds[0:n]) for n in range(1,N+1)])
    upper_bound = np.array([beta*(1-i/N) for i in range(N)])
    # plt.plot(np.arange(N), L1_norms)
    # plt.plot(np.arange(N), max_L1_norms)
    # plt.plot(np.arange(N), upper_bound)
    # plt.show()
    return np.max(np.where(max_norm_bounds <= upper_bound)[0])
    # return np.max(np.where(max_L1_norms <= upper_bound)[0])


def test_ID_focus_set():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting.suggest_act_frac = 0.45
    # setting = rb_settings.Gast20Example2()
    # setting = rb_settings.NonSAExample()
    # setting = rb_settings.ExampleFromFile("setting_data/random-size-3-uniform-(0)")
    N = 1000
    act_frac = setting.suggest_act_frac
    beta = min(act_frac, 1-act_frac)
    rb_settings.print_bandit(setting)

    T = 1200
    T_ahead = 100
    init_method = "bad" # "random" or "bad

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]

    if init_method == "random":
        init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
    elif init_method == "bad":
        init_states = np.random.choice(np.arange(4, setting.sspa_size), N, replace=True)
    else:
        raise NotImplementedError
    # init_states = 4*np.ones((N,))
    # init_states[0:int(N/3)] = 5
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("2*lambda_W = ", 2*np.linalg.norm(W, ord=2))

    id_policy = IDPolicy(setting.sspa_size, y, N, act_frac)
    setexp_policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac, W=W)

    total_reward = 0
    ideal_acts = []
    states_trace = []
    for t in range(T):
        cur_states = rb.get_states()
        actions, num_ideal_act = id_policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        states_trace.append(cur_states)
        ideal_acts.append(num_ideal_act)

    ideal_acts_lookahead_min = []
    for i in range(T):
        ideal_acts_lookahead_min.append(min(ideal_acts[i:(min(i+T_ahead,T))]))

    L1_norm_terms = []
    W_norm_terms = []
    budget_window_sizes = []
    for i in range(T):
        XD = states_to_scaled_state_counts(setting.sspa_size, N, states_trace[i][0:ideal_acts_lookahead_min[i]])
        m = ideal_acts_lookahead_min[i]/N
        xmmu_diff = XD - m*analyzer.state_probs
        L1_norm_terms.append(np.linalg.norm(xmmu_diff, ord=1)/2)
        W_norm_terms.append(np.sqrt(np.matmul(np.matmul(xmmu_diff.T, W), xmmu_diff))/2 * setexp_policy.ratiocw)
        budget_window_sizes.append(beta*(1-m))
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(T), L1_norm_terms, label="L1 norm term")
    plt.plot(np.arange(T), W_norm_terms, label="W norm term")
    plt.plot(np.arange(T), budget_window_sizes, label="budget window")
    plt.legend()
    plt.ylabel("budget-deviation bounds")
    plt.xlabel("T")
    plt.title("{}, N={}, T_ahead={}, init method={}".format("eight-states example", N, T_ahead, init_method))
    plt.savefig("figs2/ID-focus-set-compare/{}-{}-N-{}-T-{}-T_ahead-{}-init-{}".format("norm-bounds", "eight-states", N, T, T_ahead, init_method))
    plt.show()

    # L1_focus_set_sizes = []
    # W_focus_set_sizes = []
    max_L1_focus_set_sizes = []
    max_W_focus_set_sizes = []
    for i in range(T):
        # focus_set, _ = setexp_policy.get_new_focus_set(states_trace[i], np.array([]))
        # L1_focus_set_sizes.append(len(focus_set))
        # focus_set, _ = setexp_policy.get_new_focus_set(states_trace[i], np.array([]), subproblem="W")
        # LW_focus_set_sizes.append(len(focus_set))
        max_L1_focus_set_sizes.append(get_ID_max_norm_focus_set(states_trace[i], beta, analyzer.state_probs, norm="L1"))
        max_W_focus_set_sizes.append(get_ID_max_norm_focus_set(states_trace[i], beta, analyzer.state_probs, norm="W",
                                                                W=W, ratiocw=setexp_policy.ratiocw))
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(T), np.array(max_L1_focus_set_sizes)/N, label="max L1 focus set")
    plt.plot(np.arange(T), np.array(max_W_focus_set_sizes)/N, label="max W norm focus set")
    plt.plot(np.arange(T), np.array(ideal_acts_lookahead_min)/N, label="ID focus set")
    plt.legend()
    plt.ylabel("set size / N")
    plt.xlabel("T")
    plt.title("{}, N={}, T_ahead={}, init method={}".format("eight-states example", N, T_ahead, init_method))
    plt.savefig("figs2/ID-focus-set-compare/{}-{}-N-{}-T-{}-T_ahead-{}-init-{}".format("focus-set-sizes", "eight-states", N, T, T_ahead, init_method))
    plt.show()


def animate_ID_policy():
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting.suggest_act_frac = 0.45
    # setting = rb_settings.Gast20Example2()
    # setting = rb_settings.NonSAExample()
    # setting = rb_settings.ExampleFromFile("setting_data/random-size-3-uniform-(0)")
    N = 100
    act_frac = setting.suggest_act_frac
    beta = min(act_frac, 1-act_frac)
    rb_settings.print_bandit(setting)

    T = 1000
    T_ahead=100
    init_method = "bad" # "random" or "bad

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]

    if init_method == "random":
        init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
    elif init_method == "bad":
        init_states = np.random.choice(np.arange(4, setting.sspa_size), N, replace=True)
    else:
        raise NotImplementedError
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("2*lambda_W = ", 2*np.linalg.norm(W, ord=2))

    id_policy = IDPolicy(setting.sspa_size, y, N, act_frac)
    # setexp_policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac, W=W)

    total_reward = 0
    states_trace = []
    ideal_acts = []
    for t in range(T):
        cur_states = rb.get_states()
        actions, num_ideal_act = id_policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        states_trace.append(cur_states)
        ideal_acts.append(num_ideal_act)

    ideal_acts_lookahead_min = []
    for i in range(T):
        ideal_acts_lookahead_min.append(min(ideal_acts[i:(min(i+T_ahead,T))]))


    fig, ax = plt.subplots()
    upper_bound = np.array([beta*(1-i/N) for i in range(N)])
    line1 = ax.plot(np.arange(N), upper_bound, label="budget window")[0]
    line2 = ax.plot(np.arange(N), [1]*N, label="max L1 norm bound")[0]
    line3 = ax.plot(np.arange(N), [1]*N, label="L1 norm bound")[0]
    line4 = ax.axvline(ideal_acts_lookahead_min[0], 0, 1, label="actual focus set")
    ax.set_ylim([0,0.6])
    ax.set_xlabel("subsystem size")
    ax.legend()

    def update_ID_curve(frame):
        cur_states = states_trace[frame]
        norm_bounds = []
        for n in range(1,N+1):
            XD = states_to_scaled_state_counts(setting.sspa_size, N, cur_states[0:n])
            norm_bounds.append(np.linalg.norm(XD - n/N*analyzer.state_probs, ord=1) / 2)
        norm_bounds = np.array(norm_bounds)
        max_norm_bounds = np.array([np.max(norm_bounds[0:n]) for n in range(1,N+1)])
        line2.set_ydata(max_norm_bounds)
        line3.set_ydata(norm_bounds)
        line4.set_xdata(ideal_acts_lookahead_min[frame])
        ax.set_title("{}, N={},T_ahead={}, init method={}; t={}".format("eight-states example", N, T_ahead, init_method, frame))

    ani = animation.FuncAnimation(fig=fig, func=update_ID_curve, frames=400, interval=50)
    ani.save("figs2/ID-animation/{}-{}-N-{}-T-{}-T_ahead-{}-init-{}.html".format("animation", "eight-states", N, T, T_ahead, init_method),
             writer="html")
    # plt.show()



np.random.seed(114514)
np.set_printoptions(precision=5)
test_run_policies()
# edit_data()
# test_ID_focus_set()
# animate_ID_policy()

