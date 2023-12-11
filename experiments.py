import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from time import time
import rb_settings
from discrete_RB import *
import os
import pickle

### The next four functions simulate one data point for each policy.

def Priority_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=False, init_states=None, full_trace=False):
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)
    policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)

    if full_trace:
        trace = []

    tic = time()
    total_reward = 0
    for t in range(T):
        #print("state counts:", rb.get_s_counts())
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        if full_trace:
            cur_sa_freq = sa_list_to_freq(rb.sspa_size, cur_states, actions)
            # print(cur_sa_freq)
            trace.append(cur_sa_freq)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        # if True:#N == 1000:
        # print(rb.get_s_fracs())
        # if t > 800:
        #     exit()
    toc = time()
    rb_avg_reward = total_reward / T
    if verbose:
        print("Priority policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))
        #print("Running time = {} sec".format(toc-tic))

    if not full_trace:
        return rb_avg_reward
    else:
        return trace


def Priority_meanfield_experiment_one_point(T, act_frac, setting, priority_list, verbose=False, init_state_fracs=None, full_trace=False):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, init_state_fracs)
    policy = PriorityPolicy(setting.sspa_size, priority_list, N=None, act_frac=act_frac)

    if full_trace:
        trace = []

    total_reward = 0
    tic = time()
    for t in range(T):
        cur_state_fracs = mfrb.get_state_fracs()
        #print(cur_state_fracs)
        sa_pair_fracs = policy.get_sa_pair_fracs(cur_state_fracs)
        if full_trace:
            # print(sa_pair_fracs)
            trace.append(sa_pair_fracs)
        instant_reward = mfrb.step(sa_pair_fracs)
        total_reward += instant_reward
    toc = time()
    mfrb_avg_reward = total_reward / T
    if verbose:
        print("Discrete-time MF RB, {} time steps.".format(T))
        print("Average reward = {}".format(mfrb_avg_reward))
        #print("Running time = {} sec".format(toc-tic))

    if not full_trace:
        return mfrb_avg_reward
    else:
        return trace


def RandomTBPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=None, full_trace=False):
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
    policy = RandomTBPolicy(setting.sspa_size, y=y, N=N, act_frac=act_frac)

    if full_trace:
        trace = []

    total_reward = 0
    for t in range(T):
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        if full_trace:
            cur_sa_freq = sa_list_to_freq(rb.sspa_size, cur_states, actions)
            # print(cur_sa_freq)
            trace.append(cur_sa_freq)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        # if t > 200:
        #     exit()
    rb_avg_reward = total_reward / T
    if verbose:
        print("Directly Random policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))

    if not full_trace:
        return rb_avg_reward
    else:
        return trace


def FTVAPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=None, init_virtual=None, tb=None, tb_param=None, full_trace=False):
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)
    policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N, act_frac=act_frac, init_virtual=init_virtual)
    if full_trace:
        trace = []
    total_reward = 0
    tic = time()
    for t in range(T):
        prev_state = rb.get_states() # rb.states.copy() # need to copy so that no bug
        actions, virtual_actions = policy.get_actions(prev_state, tb, tb_param)
        if full_trace:
            cur_sa_freq = sa_list_to_freq(rb.sspa_size, prev_state, actions)
            # print(cur_sa_freq)
            trace.append(cur_sa_freq)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        new_state = rb.get_states()
        policy.virtual_step(prev_state, new_state, actions, virtual_actions)
    rb_avg_reward = total_reward / T
    toc = time()
    if verbose:
        if tb is None:
            print("Simulation-based policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))
        else:
            print("Simulation-based policy with tb = {}; {} arms and {} time steps. Average reward = {}".format(tb, N, T, rb_avg_reward))

    if not full_trace:
        return rb_avg_reward
    else:
        return trace



def Figure_1_experiments(init_method, need_DRP=False):
    """
    Experiments on the setting of Figure 1, which is a counterexample that does not satisfy UGAP.
    More counterexamples are stored in the file rb_settings.py.
    :param init_method: a string to be either "global" or anything else.
                        If init_method = "global", the initial states of the arms will be randomly sampled such that their
                        empirical distributions are uniformly distributed in the space of all distributions on the state space.
                        If init_method is anything else, the RBs will be initialized using the default method of the RB
                        class, i.e., all arms have state 0
    :param need_DRP: True or False. It determines whether we want to simulate the random tie-breaking policy.
    :return: None. However, the simunlation results will be stored.
    """
    settings = {"Figure1example": rb_settings.Gast20Example2()}  # modify this dictionary to run more experiments.
    # The range of N for the data points
    N_low = 100
    N_high = 1100
    # The step size of N for the data points.
    if N_low < 100:
        N_step = 10
    elif N_low < 1000:
        N_step = 100
    elif N_low < 10000:
        N_step = 1000
    else:
        N_step = 10000
    Ns = np.arange(N_low, N_high, N_step)

    num_reps = 20
    T = 4000

    for name, setting in settings.items():
        print("Simulating {}".format(name))
        act_frac = setting.suggest_act_frac # setting alpha
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=act_frac)
        priority_list, indexable = analyzer.solve_whittles_policy()
        if not indexable:
            raise ValueError("the setting is not indexable")
        mf_opt_value, y = analyzer.solve_lp()
        mfrb_avg_reward = Priority_meanfield_experiment_one_point(T, act_frac, setting, priority_list, verbose=True)
        wip_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
        drp_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
        sp_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
        for rep in range(num_reps):
            if init_method == "global":
                init_state_fracs = np.random.uniform(0, 1, size=(setting.sspa_size,))
                init_state_fracs = init_state_fracs / np.sum(init_state_fracs)
            else:
                init_state_fracs =None
            print(init_state_fracs)
            for i_th_point, N in enumerate(Ns):
                print("N = {}, rep id = {}".format(N, rep))
                init_states = np.zeros((N, ), dtype=np.int64)
                if init_method == "global":
                    for s in range(setting.sspa_size):
                        start_ind = int(N * np.sum(init_state_fracs[0:s]))
                        end_ind = int(N * np.sum(init_state_fracs[0:(s+1)]))
                        init_states[start_ind: end_ind] = s
                else:
                    init_states = None
                wip_avg_reward = Priority_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=True, init_states=init_states)
                wip_avg_rewards[rep, i_th_point] = wip_avg_reward
                if need_DRP:
                    drp_avg_reward = RandomTBPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, init_states=init_states)
                    drp_avg_rewards[rep, i_th_point] = drp_avg_reward
                sp_avg_reward = FTVAPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, init_states=init_states)
                sp_avg_rewards[rep, i_th_point] = sp_avg_reward

            # save the data every replication.
            # Some of the keys involve outdated policy names, which I have commented out.
            with open("fig_data/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method), 'wb') as f:
                setting_and_data = {
                    "num_reps": num_reps,
                    "T": T,
                    "act_frac": act_frac,
                    "Ns": Ns,
                    "name": name,
                    "setting": setting,
                    "init_method": init_method,
                    "init_state_fracs": init_state_fracs,
                    "wip_avg_rewards": wip_avg_rewards,
                    "drp_avg_rewards": drp_avg_rewards,  # drp is the random tie-breaking policy
                    "sp_avg_rewards": sp_avg_rewards,  # sp is the ftva policy
                    "priority_list": priority_list,
                    "y": y,
                    "mf_opt_value": mf_opt_value,
                    "mfrb_avg_reward": mfrb_avg_reward
                }
                pickle.dump(setting_and_data, f)


def Figure_2_experiments(init_method):
    """
    Experiments on Figure 2
    :param init_method: init_method = "bad", "random", or "evenly" or "global".
        "bad" concentrate on state 1 and 2; "random" independently uniformly sample each arm;
        "evenly" makes sure each states takes exactly 1/8 fraction (up to integer effect);
        "global" uniformly sample an initial empirical distribution for each replication
    :return: None. However, the simunlation results will be stored.
    """
    sspa_size = 8
    setting_code = "eg4action-gap-tb"
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(setting_code, sspa_size)
    setting = rb_settings.ConveyorExample(sspa_size, probs_L, probs_R, action_script, suggest_act_frac)
    rb_settings.print_bandit(setting)

    num_reps = 20
    N_low = 100
    N_high = 1100
    Ns = np.arange(N_low, N_high, 100)
    T = 20000
    act_frac = setting.suggest_act_frac

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    mf_opt_value, y = analyzer.solve_lp()
    # we manually fix the optimal dual variable to be 0
    priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
    print("priority list = ", priority_list)

    lp_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
    drp_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
    sp_avg_rewards = np.nan * np.empty((num_reps, len(Ns)))
    for rep in range(num_reps):
        if init_method == "bad":
            init_state_fracs = np.zeros((sspa_size,))
            init_state_fracs[1] = 1/3
            init_state_fracs[2] = 2/3
        elif init_method in ["evenly", "random"]:
            init_state_fracs = np.ones((sspa_size,)) / sspa_size
        elif init_method == "global":
            init_state_fracs = np.random.uniform(0, 1, size=(sspa_size,))
            init_state_fracs = init_state_fracs / np.sum(init_state_fracs)
        else:
            raise NotImplementedError
        mfrb_avg_reward = Priority_meanfield_experiment_one_point(T, act_frac, setting, priority_list, verbose=False, init_state_fracs=init_state_fracs)
        print("init state fracs = ", init_state_fracs)
        print(sum(init_state_fracs))
        print("mean field limit = ", mfrb_avg_reward)
        for i_th_point, N in enumerate(Ns):
            print("N = {}, rep id = {}".format(N, rep))
            init_states = np.zeros((N, ), dtype=np.int64)
            if init_method == "bad":
                init_states[0:int(N/3)] = 1
                init_states[int(N/3):N] = 2
            elif init_method == "random":
                init_states = np.random.choice(np.arange(0, sspa_size), N, replace=True)
            elif init_method == "evenly":
                for i in range(N):
                    init_states[i] = i % sspa_size
            elif init_method == "global":
                for s in range(sspa_size):
                    start_ind = int(N * np.sum(init_state_fracs[0:s]))
                    end_ind = int(N * np.sum(init_state_fracs[0:(s+1)]))
                    init_states[start_ind: end_ind] = s
            else:
                raise NotImplementedError
            reward = RandomTBPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, init_states=init_states)
            drp_avg_rewards[rep, i_th_point] = reward
            reward = FTVAPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, init_states=init_states)
            sp_avg_rewards[rep, i_th_point] = reward
            reward = Priority_experiment_one_point(N, T, act_frac, setting, priority_list=priority_list, verbose=True, init_states=init_states)
            lp_avg_rewards[rep, i_th_point] = reward

        name = "Figure2example"
        with open("fig_data/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method), 'wb') as f:
            setting_and_data = {
                "num_reps": num_reps,
                "T": T,
                "act_frac": act_frac,
                "Ns": Ns,
                "name": name,
                "setting_code": setting_code,
                "init_method": init_method,
                "init_state_fracs": init_state_fracs,
                "setting": setting,
                "lp_avg_rewards": lp_avg_rewards,
                "drp_avg_rewards": drp_avg_rewards,
                "sp_avg_rewards": sp_avg_rewards,
                "priority_list": priority_list,
                "y": y,
                "mf_opt_value": mf_opt_value,
                "mfrb_avg_reward": mfrb_avg_reward
            }
            pickle.dump(setting_and_data, f)


def compare_simu_tie_breakings():
    """
    On the three examples, compare the performance of:
    wip
    drp
    FTVA with "goodness" tie breaking rule (default)
    FTVA with "priority" tie breaking rule
    FTVA with "goodness-priority: tie breaking rule
    """
    settings = {"Example1": rb_settings.Gast20Example1(),
                "Example2": rb_settings.Gast20Example2(),
                "Example3": rb_settings.Gast20Example3()}
    N_low = 10000
    N_high = 100000
    if N_low < 100:
        N_step = 10
    elif N_low < 1000:
        N_step = 100
    elif N_low < 10000:
        N_step = 1000
    else:
        N_step = 10000
    Ns = np.arange(N_low, N_high, N_step)
    tbs = ["goodness", "priority", "goodness-priority"]

    #Ns = np.arange(5000, 10000, 1000)
    #Ns = np.arange(500, 1000, 100)
    T = 1000
    act_frac = 0.4

    for name, setting in settings.items():
        print("Simulating {}".format(name))
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
        priority_list, indexable = analyzer.solve_whittles_policy()
        if not indexable:
            raise ValueError("the setting is not indexable")
        mf_opt_value, y = analyzer.solve_lp()
        mfrb_avg_reward = Priority_meanfield_experiment_one_point(T, act_frac, setting, priority_list, verbose=True)
        wip_avg_rewards = []
        drp_avg_rewards = []
        sp_avg_rewards_different_tbs = {tb: [] for tb in tbs}
        for N in Ns:
            wip_avg_reward = Priority_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=True)
            wip_avg_rewards.append(wip_avg_reward)
            drp_avg_reward = RandomTBPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True)
            drp_avg_rewards.append(drp_avg_reward)
            for tb in tbs:
                sp_avg_reward = FTVAPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, tb=tb, tb_param=priority_list)
                sp_avg_rewards_different_tbs[tb].append(sp_avg_reward)

        mfrb_curve = np.array([mfrb_avg_reward]*len(Ns))
        mf_opt_curve = np.array([mf_opt_value]*len(Ns))
        fig = plt.figure()
        plt.title("Average reward of policies in {}.\n T={}, activate fraction={}".format(name, T, act_frac))
        plt.plot(Ns, wip_avg_rewards, label="WIP")
        plt.plot(Ns, drp_avg_rewards, label="DRP")
        for tb, sp_avg_rewards in sp_avg_rewards_different_tbs.items():
            plt.plot(Ns, sp_avg_rewards, label="Simu Policy, {}".format(tb))
        plt.plot(Ns, mfrb_curve, label="limit cycle")
        plt.plot(Ns, mf_opt_curve, label="mean field optimal")
        plt.xlabel("N")
        plt.ylabel("avg reward")
        plt.legend()
        plt.savefig("figs/compare-tbs-{}-N{}-{}".format(name, N_low, N_high))
        #plt.show()

def figure_from_file(name, N_low, N_high, init_method, yrange, lgloc, need_DRP):
    """
    This function helps produce Figure 1 and 2.
    :param fname, N_low, N_high, init_method: these are all parameters of the simulation, used to find the pickle file
                        saved from "Figure_1_experiments" or "Figure_2_experiments". In particular, fname is the settign name,
                        N_low and N_high are the range of the simulation, init_method is the initialization method.
    :param yrange: the range of the y-axis to show in the figure
    :param lgloc: the position of the legend
    :param need_DRP: whether we want to show curves of the random tie-breaking policy
    :return: None. But the files will be saved to the figs folder.
    """
    with open("fig_data/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method), 'rb') as f:
        setting_and_data = pickle.load(f)

    has_wip = "wip_avg_rewards" in setting_and_data

    num_reps = setting_and_data["num_reps"]
    T = setting_and_data["T"]
    act_frac = setting_and_data["act_frac"]
    Ns = setting_and_data["Ns"]
    name = setting_and_data["name"]
    setting = setting_and_data["setting"]
    if has_wip:
        wip_avg_rewards = setting_and_data["wip_avg_rewards"]
    else:
        lp_avg_rewards = setting_and_data["lp_avg_rewards"]
    drp_avg_rewards = setting_and_data["drp_avg_rewards"]
    sp_avg_rewards = setting_and_data["sp_avg_rewards"]
    priority_list = setting_and_data["priority_list"]
    y = setting_and_data["y"]
    mf_opt_value = setting_and_data["mf_opt_value"]
    mfrb_avg_reward = setting_and_data["mfrb_avg_reward"]

    fig = plt.figure(constrained_layout=True)

    num_reps_trues = np.count_nonzero(~np.isnan(sp_avg_rewards), axis=0)
    # FTVA
    sp_means = np.nanmean(sp_avg_rewards, axis=0)
    sp_yerr = 1.96 * np.nanstd(sp_avg_rewards, axis=0) / np.sqrt(num_reps_trues)
    plt.errorbar(Ns, sp_means, yerr=sp_yerr, label=r"Our policy: FTVA($\overline{\pi}^*$)", marker="o", color="r")
    # random tie-breaking policy
    if need_DRP:
        drp_means = np.nanmean(drp_avg_rewards, axis=0)
        drp_yerr = 1.96 * np.nanstd(drp_avg_rewards, axis=0) / np.sqrt(num_reps_trues)
        plt.errorbar(Ns, drp_means, yerr=drp_yerr, label="Random tie-breaking", marker="v", color="g")
    # Whittles index policy or LP-Priority
    if has_wip:
        wip_means = np.nanmean(wip_avg_rewards, axis=0)
        wip_yerr = 1.96 * np.nanstd(wip_avg_rewards, axis=0) / np.sqrt(num_reps_trues)
        plt.errorbar(Ns, wip_means, yerr=wip_yerr, label="Whittle Index/LP-Priority", marker="x", color="b")
    else:
        lp_means = np.nanmean(lp_avg_rewards, axis=0)
        lp_yerr = 1.96 * np.nanstd(lp_avg_rewards, axis=0) / np.sqrt(num_reps_trues)
        plt.errorbar(Ns, lp_means, yerr=lp_yerr, label="LP-Priority", marker="x", color="b")
    #limit cycle and optimal value
    mfrb_curve = np.array([mfrb_avg_reward]*len(Ns))
    mf_opt_curve = np.array([mf_opt_value]*len(Ns))
    plt.plot(Ns, mf_opt_curve, label="Upper bound", linestyle="--", color="black")
    #plt.plot(Ns, mfrb_curve, linestyle="-.", color="purple") # "Mean field limit", but we don't show label

    plt.rcParams.update({'font.size': 20})
    #plt.title("Average reward of policies in {}.\n T={}, {}={}".format(name, T, r'$\alpha$', act_frac))
    plt.xlabel("N", fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel("Avg Reward", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(yrange)
    plt.grid()
    plt.legend(loc=lgloc)
    plt.savefig("figs/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method))
    plt.show()

def figure_from_file_multiple_inits(name, N_low, N_high, init_method, yrange, lgloc, plot_type, need_DRP):
    """
    This function helps produce Figure 6.
    When the empirical distribution of the initial states are different across sample paths, we use this function to
    show the variations of the performance.
    There can be three types of plot, by setting "plot_type" to be one of "show-all", "bar", or "max-min".
    """
    with open("fig_data/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method), 'rb') as f:
        setting_and_data = pickle.load(f)

    has_wip = "wip_avg_rewards" in setting_and_data

    num_reps = setting_and_data["num_reps"]
    T = setting_and_data["T"]
    act_frac = setting_and_data["act_frac"]
    Ns = setting_and_data["Ns"]
    name = setting_and_data["name"]
    setting = setting_and_data["setting"]
    if has_wip:
        wip_avg_rewards = setting_and_data["wip_avg_rewards"]
    else:
        lp_avg_rewards = setting_and_data["lp_avg_rewards"]
    drp_avg_rewards = setting_and_data["drp_avg_rewards"]
    sp_avg_rewards = setting_and_data["sp_avg_rewards"]
    priority_list = setting_and_data["priority_list"]
    y = setting_and_data["y"]
    mf_opt_value = setting_and_data["mf_opt_value"]
    mfrb_avg_reward = setting_and_data["mfrb_avg_reward"]

    # initialize figure
    fig = plt.figure(constrained_layout=True)
    if plot_type == "show-all":
        for i in range(num_reps):
            cur_label = r"Our policy: FTVA($\overline{\pi}^*$)" if i==0 else "_"
            plt.plot(Ns, sp_avg_rewards[i,:], label=cur_label, marker="o", color="r")
            # random tie-breaking policy
            cur_label = "Random tie-breaking" if i==0 else "_"
            plt.plot(Ns, drp_avg_rewards[i,:], label=cur_label, marker="v", color="g")
             # Whittles index policy or LP-Priority
            if has_wip:
                cur_label = "Whittle Index/LP-Priority" if i==0 else "_"
                plt.plot(Ns, wip_avg_rewards[i,:], label=cur_label, marker="x", color="b")
            else:
                cur_label = "LP-Priority" if i==0 else "_"
                plt.plot(Ns, lp_avg_rewards[i,:], label=cur_label, marker="x", color="b")
            # optimal value
            cur_label = "Upper bound" if i==0 else "_"
            mf_opt_curve = np.array([mf_opt_value]*len(Ns))
            plt.plot(Ns, mf_opt_curve, label=cur_label, linestyle="--", color="black")
    elif plot_type == "bar":
        xs = np.arange(len(Ns))
        cur_max = np.max(sp_avg_rewards, axis=0)
        cur_min = np.min(sp_avg_rewards, axis=0)
        plt.bar(xs, height=cur_max-cur_min, bottom=cur_min, label=r"Our policy: FTVA($\overline{\pi}^*$)", color="r")
        cur_max = np.max(drp_avg_rewards, axis=0)
        cur_min = np.min(drp_avg_rewards, axis=0)
        plt.bar(xs, height=cur_max-cur_min, bottom=cur_min, label="Random tie-breaking", color="g")
        if has_wip:
            cur_max = np.max(wip_avg_rewards, axis=0)
            cur_min = np.min(wip_avg_rewards, axis=0)
            plt.bar(xs, height=cur_max-cur_min, bottom=cur_min, label="Whittle Index/LP-Priority", color="b")
        else:
            cur_max = np.max(lp_avg_rewards, axis=0)
            cur_min = np.min(lp_avg_rewards, axis=0)
            plt.bar(xs, height=cur_max-cur_min, bottom=cur_min, label="LP-Priority", color="b")
        mf_opt_curve = np.array([mf_opt_value]*len(Ns))
        plt.plot(xs, mf_opt_curve, label="Upper bound", linestyle="--", color="black")
        plt.xticks(ticks=xs[0:-1:2], labels=Ns[0:-1:2])
    elif plot_type == "max-min":
        cur_min = np.min(sp_avg_rewards, axis=0)
        plt.plot(Ns,cur_min, label=r"FTVA($\overline{\pi}^*$), min of 20 sets", color="r", marker="o")
        if need_DRP:
            cur_max = np.max(drp_avg_rewards, axis=0)
            plt.plot(Ns, cur_max, label="Random tie-breaking, max of 20 sets", color="g", marker="v")
        if has_wip:
            cur_max = np.max(wip_avg_rewards, axis=0)
            plt.plot(Ns, cur_max, label="Whittle Index/LP-Priority, max of 20 sets", color="b", marker="x")
        else:
            cur_max = np.max(lp_avg_rewards, axis=0)
            plt.plot(Ns, cur_max, label="LP-Priority, max of 20 sets", color="b", marker="x")
        mf_opt_curve = np.array([mf_opt_value]*len(Ns))
        plt.plot(Ns, mf_opt_curve, label="Upper bound", linestyle="--", color="black")
    else:
        raise NotImplementedError

    # figure settings
    plt.rcParams.update({'font.size': 15})
    #plt.rcParams['text.usetex'] = True
    #plt.title("Average reward of policies in {}.\n T={}, {}={}".format(name, T, r'$\alpha$', act_frac))
    plt.xlabel("N", fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel("Avg Reward", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(yrange)
    plt.grid()
    plt.legend(loc=lgloc)
    #plt.savefig("figs/Formal-{}-N{}-{}-{}".format(name, N_low, N_high, init_method))
    plt.show()


def visualize_states_actions_Figure_2_example(no_title):
    """
    This function runs a short trace and visualize the states and actions for the RB example in Figure 2.
    It produces figures 3, 7 and 8.
    :return:
    """
    # defining the RB example
    sspa_size = 8
    setting_code = "eg4action-gap-tb"
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(setting_code, sspa_size)
    setting = rb_settings.ConveyorExample(sspa_size, probs_L, probs_R, action_script, suggest_act_frac)
    rb_settings.print_bandit(setting)

    # fix N, T and alpha
    N = 500
    T = 500
    act_frac = setting.suggest_act_frac

    # initialization
    init_method = "bad"
    if init_method == "bad":
        init_state_fracs = np.zeros((sspa_size,))
        init_state_fracs[1] = 1/3
        init_state_fracs[2] = 2/3
    elif init_method in ["evenly", "random"]:
        init_state_fracs = np.ones((sspa_size,)) / sspa_size
    elif init_method == "global":
        init_state_fracs = np.random.uniform(0, 1, size=(sspa_size,))
        init_state_fracs = init_state_fracs / np.sum(init_state_fracs)
    else:
        raise NotImplementedError
    print("init state fracs = ", init_state_fracs)
    init_states = np.zeros((N, ), dtype=np.int64)
    if init_method == "bad":
        init_states[0:int(N/3)] = 1
        init_states[int(N/3):N] = 2
    elif init_method == "random":
        init_states = np.random.choice(np.arange(0, sspa_size), N, replace=True)
    elif init_method == "evenly":
        for i in range(N):
            init_states[i] = i % sspa_size
    elif init_method == "global":
        for s in range(sspa_size):
            start_ind = int(N * np.sum(init_state_fracs[0:s]))
            end_ind = int(N * np.sum(init_state_fracs[0:(s+1)]))
            init_states[start_ind: end_ind] = s
    else:
        raise NotImplementedError

    # solve the LP
    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    mf_opt_value, y = analyzer.solve_lp()
    # we manually fix the optimal dual variable to be 0
    priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
    print("priority list = ", priority_list)

    # generate a short trace for each policy
    traces = []
    trace = RandomTBPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=init_states, full_trace=True)
    traces.append(trace)
    trace = Priority_experiment_one_point(N, T, act_frac, setting, priority_list=priority_list, verbose=False, init_states=init_states, full_trace=True)
    traces.append(trace)
    trace = FTVAPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=init_states, init_virtual=None, full_trace=True)
    traces.append(trace)
    for k in range(len(traces)):
        traces[k] = np.stack(traces[k])
        print(traces[k])

    ## Generate Figure 7(a), where we visualize the state evolution for a larger time scale (500 time slots) under three policies
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(9,5))
    for k in range(3):
        state_vs_time = np.sum(traces[k], axis=2).transpose((1,0))
        axs[k].imshow(state_vs_time, origin="lower", aspect="auto", cmap="summer")
        axs[k].set_yticks(ticks=np.arange(sspa_size)+0.5, labels=[r'${}$'.format(i) for i in np.arange(sspa_size)])
        for label in axs[k].get_yticklabels():
            label.set_verticalalignment('top')
        axs[k].set_ylabel("state", fontsize=15)
    axs[1].set_xticks(ticks=np.append(np.arange(0, T, 100), T-1),
                           labels=[r'${}$'.format(i) for i in np.append(np.arange(0, T, 100), T-1)])

    plt.xlabel("time slot", fontsize=15)
    if not no_title:
        fig.suptitle("Fraction of Arms in Each State\n"+"Top to Bottom: Random Tie-breaking, LP Priority, and FTVA.", fontsize=15)
        plt.subplots_adjust(left=0.05, top=0.89, bottom=0.1, hspace=0.06, right=0.91)
        cax = plt.axes([0.93, 0.1, 0.03, 0.8])
    else:
        plt.subplots_adjust(left=0.05, top=0.99, bottom=0.1, hspace=0.06, right=0.91)
        cax = plt.axes([0.93, 0.1, 0.03, 0.87])
    # set color bar
    cmap = mpl.cm.summer
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', location='right')
    cax.set_yticks(np.arange(0,1+1/8,1/8), [r'$0$']+[r'${}/8$'.format(i) for i in np.arange(1,8)]+[r'$1$'])
    if not no_title:
        plt.savefig('figs/large_time_scale_state_time.png', dpi=1000)
    else:
        plt.savefig('figs/large_time_scale_state_time-notitle.png', dpi=1000)


    ### Generate figure 3, 7(b) and 8
    benchmark_names = ["Random Tie-breaking", "LP-Priority"]
    start_t = 250
    burn_in_t = 0
    obs_t = 40
    for k in range(2):
        # run the FTVA from the middle of Random Tie-breaking or LP-Priority's sample path
        trace = traces[k]
        init_state_fracs = np.sum(trace[start_t,:,:], axis=1)
        init_states = states_from_state_fracs(sspa_size, N, init_state_fracs)
        trace_ftva = FTVAPolicy_experiment_one_point(N, burn_in_t + obs_t, act_frac, setting, y, verbose=False, init_states=init_states, init_virtual=None, full_trace=True)
        trace_ftva = np.stack(trace_ftva)

        ### generate Figure 3 and 7(b)
        # prepare the heatmaps of state evolutions
        state_vs_time = np.sum(trace, axis=2).transpose((1,0))
        state_vs_time_ftva = np.sum(trace_ftva, axis=2).transpose((1,0))

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(9,5))
        axs[0].imshow(state_vs_time[:, (start_t+burn_in_t):(start_t+burn_in_t+obs_t)], origin="lower", aspect="auto", cmap="summer")
        axs[1].imshow(state_vs_time_ftva[:, burn_in_t:], origin="lower", aspect="auto", cmap="summer")

        # prepare for the vector fields of drifts
        trace_slice = trace[(start_t+burn_in_t):(start_t+burn_in_t+obs_t),:,:]
        drift_vs_time = np.zeros(trace_slice.shape[0:2])
        drift_vs_time[:,0:4] = setting.probs_R[0:4] * trace_slice[:,0:4,1] - setting.probs_L[0:4] * trace_slice[:,0:4,0]
        drift_vs_time[:,4:8] = setting.probs_R[4:8] * trace_slice[:,4:8,0] - setting.probs_L[4:8] * trace_slice[:,4:8,1]
        trace_slice_ftva = trace_ftva[burn_in_t:(burn_in_t+obs_t),:,:]
        drift_vs_time_ftva = np.zeros(trace_slice_ftva.shape[0:2])
        drift_vs_time_ftva[:,0:4] = setting.probs_R[0:4] * trace_slice_ftva[:,0:4,1] - setting.probs_L[0:4] * trace_slice_ftva[:,0:4,0]
        drift_vs_time_ftva[:,4:8] = setting.probs_R[4:8] * trace_slice_ftva[:,4:8,0] - setting.probs_L[4:8] * trace_slice_ftva[:,4:8,1]
        # print(drift_vs_time_ftva)

        time_mesh, state_mesh = np.meshgrid(np.arange(0, obs_t), np.arange(0, sspa_size), indexing="ij")
        horizontal_drift = np.zeros_like(time_mesh)
        # print(time_mesh[1,3], state_mesh[1,3])

        axs[0].quiver(time_mesh, state_mesh, horizontal_drift, drift_vs_time, scale=2,
                      headlength=5, headwidth=3, angles="xy", color=drift_array_to_rgb_array(drift_vs_time))
        axs[1].quiver(time_mesh, state_mesh, horizontal_drift, drift_vs_time_ftva, scale=2,
                      headlength=5, headwidth=3, angles="xy", color=drift_array_to_rgb_array(drift_vs_time_ftva))

        # set ticks and labels
        for i in range(2):
            axs[i].set_yticks(ticks=np.arange(sspa_size)+0.5, labels=[r'${}$'.format(i) for i in np.arange(sspa_size)])
            for label in axs[i].get_yticklabels():
                label.set_verticalalignment('top')
            axs[i].set_ylabel("state", fontsize=15)
        # when set x ticks, we make sure the last number is included
        axs[1].set_xticks(ticks=np.append(np.arange(0, obs_t, 4), obs_t-1),
                          labels=[r'${}$'.format(i) for i in np.append(np.arange(start_t+burn_in_t, start_t+burn_in_t+obs_t, 4), start_t+burn_in_t+obs_t-1)])
        for label in axs[1].get_xticklabels():
            label.set_horizontalalignment('center')
        plt.xlabel("time slot", fontsize=15)

        if not no_title:
            fig.suptitle("Comparing the Drifts of {} and FTVA".format(benchmark_names[k]), fontsize=15)
            plt.subplots_adjust(left=0.05, top=0.89, bottom=0.1, hspace=0.06, right=0.91)
            cax = plt.axes([0.93, 0.1, 0.03, 0.8])
        else:
            plt.subplots_adjust(left=0.05, top=0.99, bottom=0.1, hspace=0.06, right=0.91)
            cax = plt.axes([0.93, 0.1, 0.03, 0.87])

        # set color bar and save the figures
        cmap = mpl.cm.summer
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', location='right')
        cax.set_yticks(np.arange(0,1+1/8,1/8), [r'$0$']+[r'${}/8$'.format(i) for i in np.arange(1,8)]+[r'$1$'])
        #plt.tight_layout()
        #plt.show()
        if not no_title:
            plt.savefig("figs/vector-field-{}-vs-ftva.png".format(benchmark_names[k]), dpi=1000)
        else:
            plt.savefig("figs/vector-field-{}-vs-ftva-notitle.png".format(benchmark_names[k]), dpi=1000)

        ### Generate Figure 8
        obs_t_short = 4
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(5,5))
        sspa = np.arange(0, sspa_size)
        width= 0.8/obs_t_short
        wid_margin = 0.9/obs_t_short
        traces_in_this_figure = [("benchmark", trace), ("ftva",trace_ftva)]
        for i in range(2):
            policy_name = traces_in_this_figure[i][0]
            cur_trace = traces_in_this_figure[i][1]
            for t in range(obs_t_short):
                if policy_name=="benchmark":
                    sa_frac_slice = cur_trace[start_t+burn_in_t+t,:,:]
                else:
                    sa_frac_slice = cur_trace[burn_in_t+t,:,:]
                # put the action 0 on bottom, action 1 on top
                # preferred actions
                bar_height_p = np.zeros((sspa_size,))
                bar_height_p[0:4] = sa_frac_slice[0:4,1]
                bar_height_p[4:8] = sa_frac_slice[4:8,0]
                bar_bottom_p = np.zeros((sspa_size,))
                bar_bottom_p[0:4] = sa_frac_slice[0:4,0]
                # non-preferred actions
                bar_height_np = np.zeros((sspa_size,))
                bar_height_np[0:4] = sa_frac_slice[0:4,0]
                bar_height_np[4:8] = sa_frac_slice[4:8,1]
                bar_bottom_np = np.zeros((sspa_size,))
                bar_bottom_np[4:8] = sa_frac_slice[4:8,0]
                axs[i].bar(sspa+(t-obs_t_short/2+0.5)*wid_margin, bar_height_p, bottom=bar_bottom_p, width=width, color='b')
                axs[i].bar(sspa+(t-obs_t_short/2+0.5)*wid_margin, bar_height_np, bottom=bar_bottom_np, width=width, color='r')
                # legend
                blue_patch = mpatches.Patch(color='blue', label='preferred action')
                red_patch = mpatches.Patch(color='red', label='non-preferred action')
                axs[i].legend(handles=[blue_patch, red_patch], fontsize=11)

        # set ticks and labels
        for i in range(2):
            # axs[i].set_yticks(ticks=np.arange(sspa_size)+0.5, labels=[r'${}$'.format(i) for i in np.arange(sspa_size)])
            # for label in axs[i].get_yticklabels():
            #     label.set_verticalalignment('top')
            axs[i].set_ylabel("fraction of arms", fontsize=12)
        axs[1].set_xticks(ticks=np.arange(0, obs_t_short, 4),
                          labels=[r'${}$'.format(i) for i in np.arange(start_t+burn_in_t, start_t+burn_in_t+obs_t_short, 4)])
        for label in axs[1].get_xticklabels():
            label.set_horizontalalignment('left')
        plt.xticks(ticks=np.arange(0,8), labels=[r'${}$'.format(i) for i in np.arange(sspa_size)])
        plt.xlabel("state", fontsize=15)
        fig.suptitle("State-action fraction starting from time {} \n under {} and FTVA ".format(start_t+burn_in_t, benchmark_names[k], ), fontsize=12)
        plt.subplots_adjust(left=0.07, right=0.95, top=0.88, bottom=0.1, hspace=0.06)
        plt.savefig("figs/sa-fraction-{}-vs-ftva.png".format(benchmark_names[k]), dpi=1000)


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=5, suppress=True)

    if not os.path.exists("figs"):
        os.makedirs("figs")
    if not os.path.exists("fig_data"):
        os.makedirs("fig_data")

    # # Generate data for the two examples in Figure 1 and Figure 2; they will be saved to the folder "fig_data"
    # # uncomment the block of code below to rerun the simulations
    # # Figure 1 and Figure 2
    # Figure_1_experiments(init_method="default")
    # Figure_2_experiments(init_method="bad")
    # # Figure 6
    # Figure_1_experiments(init_method="global")
    # Figure_2_experiments(init_method="global")

    # Make figures. They will be saved to the folder "figs"
    # make Figure 1
    figure_from_file("Figure1Example", 100, 1100, "default", (0.093, 0.125), "lower right", need_DRP=False)
    # make Figure 2
    figure_from_file("Figure2Example", 100, 1100, "bad", (0, 0.014), "right", need_DRP=True)
    # make Figure 6
    figure_from_file_multiple_inits("Figure1Example", 100, 1100, "global", (0.093, 0.125), "lower right", plot_type="max-min", need_DRP=False)
    figure_from_file_multiple_inits("Figure2Example", 100, 1100, "global", (0, 0.021), "upper right", plot_type="max-min", need_DRP=True)
    # make Figure 3, 7, 8
    visualize_states_actions_Figure_2_example(no_title=True)
