import numpy as np
import cvxpy as cp
import scipy
import pickle
import time
from matplotlib import pyplot as plt
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
    # probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters("eg4action-gap-tb", 8)
    # setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    # setting.suggest_act_frac = 0.45
    # setting = rb_settings.Gast20Example2()
    # setting = rb_settings.NonSAExample()
    setting = rb_settings.ExampleFromFile("setting_data/random-size-3-uniform-(0)")
    N = 1000
    act_frac = setting.suggest_act_frac
    rb_settings.print_bandit(setting)

    # init_states = np.random.choice(sspa, N, replace=True)
    # rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("W=", W)
    priority = analyzer.solve_LP_Priority()
    print("LP Priority=", priority)
    U = analyzer.compute_U(abstol=1e-10)[0]
    print("U=\n", U)

    init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
    # init_states = np.random.choice(np.arange(4, setting.sspa_size), N, replace=True)
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
    for t in range(T):
        cur_states = rb.get_states()
        OL_set = twoset_policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
        actions = twoset_policy.get_actions(cur_states, OL_set)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        total_OL_set_size += len(OL_set)
        if t%100 == 0:
            sa_fracs = sa_list_to_freq(setting.sspa_size, cur_states, actions)
            s_fracs = np.sum(sa_fracs, axis=1)
            print("t={}\ns_fracs={}".format(t, s_fracs))
            print("OL set size before rounding={}, after rounding={}".format((twoset_policy.m.value or 0)*N, len(OL_set)))
    print("avg reward = {}".format(total_reward / T))
    print("avg OL set size = {}".format(total_OL_set_size / T))




np.random.seed(114514)
np.set_printoptions(precision=2)
test_run_policies()
# edit_data()



# A = np.array([1,2,9,4,5,6,7])
# print(A[-3:])
# print(A[-3-1:-1])
# print(A[-3:])
# print(np.sort(A))
# indices = np.array([3,4,5])
# A[indices] = 0
# print(A)
