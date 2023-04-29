from matplotlib import pyplot as plt
from time import time
import rb_settings
from discrete_RB import *
import os

def WIP_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=False, init_states=None):
    # budget = N*act_frac
    # assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    # budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)
    policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)

    tic = time()
    total_reward = 0
    for t in range(T):
        #print("state counts:", rb.get_s_counts())
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        # print(rb.get_s_fracs())
        # if t > 200:
        #     exit()
    toc = time()
    rb_avg_reward = total_reward / T
    if verbose:
        print("Priority policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))
        #print("Running time = {} sec".format(toc-tic))

    return rb_avg_reward


def WIP_meanfield_simulation(T, act_frac, setting, priority_list, verbose=False):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor)
    policy = PriorityPolicy(setting.sspa_size, priority_list, N=None, act_frac=act_frac)

    total_reward = 0
    tic = time()
    for t in range(T):
        cur_state_fracs = mfrb.get_state_fracs()
        #print(cur_state_fracs)
        sa_pair_fracs = policy.get_sa_pair_fracs(cur_state_fracs)
        #print(sa_pair_fracs)
        instant_reward = mfrb.step(sa_pair_fracs)
        total_reward += instant_reward
    toc = time()
    mfrb_avg_reward = total_reward / T
    if verbose:
        print("Discrete-time MF RB, {} time steps.".format(T))
        print("Average reward = {}".format(mfrb_avg_reward))
        #print("Running time = {} sec".format(toc-tic))

    return mfrb_avg_reward


def DirectRandomPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=None):
    # budget = N*act_frac
    # assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    # budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
    policy = DirectRandomPolicy(setting.sspa_size, y=y, N=N, act_frac=act_frac)

    total_reward = 0
    for t in range(T):
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        # print(rb.get_s_fracs())
        # if t > 200:
        #     exit()
    rb_avg_reward = total_reward / T
    if verbose:
        print("Directly Random policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))

    return rb_avg_reward


def SimuPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False, init_states=None, tb=None, tb_param=None):
    # budget = N*act_frac
    # assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    # budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states=init_states)
    policy = SimuPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N, act_frac=act_frac)
    total_reward = 0
    tic = time()
    for t in range(T):
        prev_state = rb.get_states() # rb.states.copy() # need to copy so that no bug
        actions, virtual_actions = policy.get_actions(prev_state, tb, tb_param)
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

    return rb_avg_reward


def test_cycle(setting, priority_list, act_frac, try_steps, eps, verbose=False):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor)
    policy = PriorityPolicy(setting.sspa_size, priority_list, act_frac=act_frac)
    #run simulation for a try_steps and see if the fraction of states converge
    is_convergent = False
    disc_error = 0
    cur_state_fracs = mfrb.get_state_fracs()
    for t in range(try_steps):
        cur_sa_pair_fracs = policy.get_sa_pair_fracs(cur_state_fracs)
        mfrb.step(cur_sa_pair_fracs)
        new_state_fracs = mfrb.get_state_fracs()
        if verbose:
            print("state_fracs = ", new_state_fracs)
        disc_error = 0.5 * disc_error + np.linalg.norm(new_state_fracs - cur_state_fracs)
        if disc_error < eps:
            is_convergent = True
            break
        else:
            cur_state_fracs = new_state_fracs

    has_cycle = not is_convergent
    return has_cycle


def search_for_WIP_counterexamples():
    num_of_settings = 10**3
    sspa_size = 4
    try_steps = 100
    act_frac = 0.4
    eps = 1e-4
    alg = "wip"  # wip or lp

    num_maybe_nonindexable = 0
    num_indexable_cycles = 0
    for k in range(num_of_settings):
        f_path = "settings/countereg-{}-{}".format(alg, k)
        setting = rb_settings.RandomExample(sspa_size, distr="beta05")
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
        if alg == "wip":
            priority_list, indexable = analyzer.solve_whittles_policy()
            if not indexable:
                num_maybe_nonindexable += 1
                print("nonindexable example found")
                continue
        elif alg == "lp":
            priority_list = analyzer.solve_LP_Priority()
        has_cycle = test_cycle(setting, priority_list, act_frac, try_steps=try_steps, eps=eps)
        if has_cycle:
            num_indexable_cycles += 1
            print("cycle found.")
            rb_settings.save_bandit(setting, f_path, other_params={"act_frac": act_frac})
            rb_settings.print_bandit(setting)

    print("fraction of nonindexable:", num_maybe_nonindexable / num_of_settings)
    print("fraction of indexable cycle", num_indexable_cycles / num_of_settings)


def see_counterexamples():
    directory = "settings_archive/d4_actfrac04_batch1e3" #"settings_archive/batch_of_1000_beta05_actfrac_04"
    alg = "wip" # wip or lp
    act_frac = 0.4
    f_names = os.listdir(directory)
    for f_name in f_names:
        f_path = directory + "/" + f_name
        print(f_path)
        setting = rb_settings.ExampleFromFile(f_path)
        rb_settings.print_bandit(setting)
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=act_frac)
        if alg == "wip":
            priority_list, indexable = analyzer.solve_whittles_policy()
        elif alg == "lp":
            priority_list = analyzer.solve_LP_Priority()
        else:
            raise NotImplementedError
        has_cycle = test_cycle(setting, priority_list, act_frac, try_steps=1000, eps=1e-4, verbose=True)
        print("has_cycle=", has_cycle)


def when_is_whittles_same_as_lp():
    for i in range(10):
        print("-----------{}------------".format(i))
        setting = rb_settings.RandomExample(3, "beta05")
        #setting = rb_settings.Gast20Example1()
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=0.4)
        lp_priority_list = analyzer.solve_LP_Priority()
        whittle_priority_list, indexable = analyzer.solve_whittles_policy()
        if not indexable:
            continue
        print("lp-priority", lp_priority_list)
        print("wip", whittle_priority_list)


### EXPERIMENTS ON EXAMPLES 1, 2, 3
def compare_policies_experiments():
    settings = {"Example1": rb_settings.Gast20Example1(),
                "Example2": rb_settings.Gast20Example2(),
                "Example3": rb_settings.Gast20Example3()}
    N_low = 100
    N_high = 1000
    if N_low < 100:
        N_step = 10
    elif N_low < 1000:
        N_step = 100
    elif N_low < 10000:
        N_step = 1000
    else:
        N_step = 10000
    Ns = np.arange(N_low, N_high, N_step)

    T = 1000
    act_frac = 0.4  # this should be 0.4. Don't change it.
    special = "mg" # maintaining good arms

    for name, setting in settings.items():
        print("Simulating {}".format(name))
        analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac=act_frac)
        priority_list, indexable = analyzer.solve_whittles_policy()
        if not indexable:
            raise ValueError("the setting is not indexable")
        mf_opt_value, y = analyzer.solve_lp()
        mfrb_avg_reward = WIP_meanfield_simulation(T, act_frac, setting, priority_list, verbose=True)
        wip_avg_rewards = []
        drp_avg_rewards = []
        sp_avg_rewards = []
        for N in Ns:
            wip_avg_reward = WIP_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=True)
            wip_avg_rewards.append(wip_avg_reward)
            drp_avg_reward = DirectRandomPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True)
            drp_avg_rewards.append(drp_avg_reward)
            sp_avg_reward = SimuPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True)
            sp_avg_rewards.append(sp_avg_reward)

        mfrb_curve = np.array([mfrb_avg_reward]*len(Ns))
        mf_opt_curve = np.array([mf_opt_value]*len(Ns))
        fig = plt.figure()
        plt.title("Average reward of policies in {}.\n T={}, activate fraction={}".format(name, T, act_frac))
        plt.plot(Ns, wip_avg_rewards, label="WIP")
        plt.plot(Ns, drp_avg_rewards, label="DRP")
        plt.plot(Ns, sp_avg_rewards, label="Simu Policy")
        plt.plot(Ns, mfrb_curve, label="limit cycle")
        plt.plot(Ns, mf_opt_curve, label="mean field optimal")
        plt.xlabel("N")
        plt.ylabel("avg reward")
        plt.legend()
        plt.savefig("figs/{}-{}-N{}-{}".format(special, name, N_low, N_high))
        #plt.show()



# EXPERIMENTS ON EXAMPLE 4
def compare_policies_conveyor_example():
    sspa_size = 8
    setting_code = "eg4archive2"
    probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(setting_code, sspa_size)
    setting = rb_settings.ConveyorExample(sspa_size, probs_L, probs_R, action_script, suggest_act_frac)
    rb_settings.print_bandit(setting)
    N_low = 100
    N_high = 1000
    Ns = np.arange(N_low, N_high+100, 100)
    T = 1000
    act_frac = setting.suggest_act_frac
    init_method = "bad"  # or "bad", "random", or "evenly"

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    opt_value, y = analyzer.solve_lp()
    priority_list = analyzer.solve_LP_Priority()
    #priority_list = [1,2,3,4,0,9,8,7,6,5]
    print("priority list = ", priority_list)
    mf_opt_curve = np.array([opt_value]*len(Ns))

    drp_rewards = []
    simup_rewards = []
    lp_priority_rewards = []
    for N in Ns:
        init_states = np.zeros((N, ), dtype=np.int64)
        if init_method == "bad":
            init_states[0:int(N/3)] = 2
            init_states[int(N/3):N] = 3
        elif init_method == "random":
            init_states = np.random.choice(np.arange(0, sspa_size), N, replace=True)
        elif init_method == "evenly":
            for i in range(N):
                init_states[i] = i % sspa_size
        else:
            raise NotImplementedError
        reward = DirectRandomPolicy_experiment_one_point(N, T,  act_frac, setting, y, verbose=True, init_states=init_states)
        drp_rewards.append(reward)
        reward = SimuPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, init_states=init_states)
        simup_rewards.append(reward)
        reward = WIP_experiment_one_point(N, T, act_frac, setting, priority_list=priority_list, verbose=True, init_states=init_states)
        lp_priority_rewards.append(reward)
    plt.figure()
    plt.title("My example. Average reward of policies in Conveyor Example.\n T={}, activate fraction={}".format(T, act_frac))
    plt.plot(Ns, drp_rewards, label="DRP")
    plt.plot(Ns, lp_priority_rewards, label="LP")
    plt.plot(Ns, simup_rewards, label="Simu_policy")
    plt.plot(Ns, mf_opt_curve, label="mf-opt")
    plt.xlabel("N")
    plt.ylabel("avg reward")
    plt.legend()
    plt.savefig("figs/{}-size{}-init_{}-N{}-{}".format(setting_code, sspa_size, init_method, N_low, N_high))
    plt.show()


# EXPERIMENTS ON TIE BREAKING RULES
def compare_simu_tie_breakings():
    """
    On the three examples,
    compare the performance of:
    wip
    drp
    SimuPolicy with "goodness" tie breaking rule (default)
    SimuPolicy with "priority" tie breaking rule
    SimuPolicy with "goodness-priority: tie breaking rule
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
        mfrb_avg_reward = WIP_meanfield_simulation(T, act_frac, setting, priority_list, verbose=True)
        wip_avg_rewards = []
        drp_avg_rewards = []
        sp_avg_rewards_different_tbs = {tb: [] for tb in tbs}
        for N in Ns:
            wip_avg_reward = WIP_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=True)
            wip_avg_rewards.append(wip_avg_reward)
            drp_avg_reward = DirectRandomPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True)
            drp_avg_rewards.append(drp_avg_reward)
            for tb in tbs:
                sp_avg_reward = SimuPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=True, tb=tb, tb_param=priority_list)
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


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=5, suppress=True)

    compare_policies_experiments()



    ### To-do list:
    ### need to have a confidence interval by repeating multiple times
    ### rewrite the priority policy part with explicit code.

    ### what is the RIGHT implementation of action-gap based tie breaking? The subsidy is not unique.
    #  0 subsidy is not convincing; other arbitrary subsidies are not good either.
    # hopefully, break DRP and LP-Priority for all legal subsidies. Possible things to try:
    # initialize in bad states 0,1,2
    # action script = 11110000 instead of 01110000 so that we can break DRP
    # decrease prob_R[0] if we use action script 01110000


    #  manually control the subsidy and do a set of experiments
    # fix the bug of not accepting fractional budget

