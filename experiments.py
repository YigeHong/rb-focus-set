from matplotlib import pyplot as plt
from time import time
import rb_settings
from discrete_RB import *
import os

def WIP_experiment_one_point(N, T, act_frac, setting, priority_list, verbose=False):
    budget = N*act_frac
    assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N)
    policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, budget=budget)

    tic = time()
    total_reward = 0
    for t in range(T):
        #print("state counts:", rb.get_s_counts())
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
    toc = time()
    rb_avg_reward = total_reward / T
    if verbose:
        print("Whittle's index policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))
        #print("Running time = {} sec".format(toc-tic))

    return rb_avg_reward


def WIP_meanfield_simulation(T, act_frac, setting, priority_list, verbose=False):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor)
    policy = PriorityPolicy(setting.sspa_size, priority_list, act_frac=act_frac)

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


def DirectRandomPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False):
    budget = N*act_frac
    assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N)
    policy = DirectRandomPolicy(setting.sspa_size, y=y, N=N, budget=budget)

    total_reward = 0
    for t in range(T):
        cur_states = rb.get_states()
        actions = policy.get_actions(cur_states)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
    rb_avg_reward = total_reward / T
    if verbose:
        print("Directly Random policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))

    return rb_avg_reward


def SimuPolicy_experiment_one_point(N, T, act_frac, setting, y, verbose=False):
    budget = N*act_frac
    assert budget.is_integer(), "we do not do the rounding, so make sure N * act_frac is an integer."
    budget = int(budget)

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N)
    policy = SimuPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, budget, y)
    total_reward = 0
    tic = time()
    for t in range(T):
        prev_state = rb.get_states() # rb.states.copy() # need to copy so that no bug
        actions, virtual_actions = policy.get_actions(prev_state)
        instant_reward = rb.step(actions)
        total_reward += instant_reward
        new_state = rb.get_states()
        policy.virtual_step(prev_state, new_state, actions, virtual_actions)
    rb_avg_reward = total_reward / T
    toc = time()
    if verbose:
        print("Simulation-based policy, {} arms and {} time steps. Average reward = {}".format(N, T, rb_avg_reward))

    return rb_avg_reward


def Compare_policies_experiments():
    settings = {"Example1": rb_settings.Gast20Example1(),
                "Example2": rb_settings.Gast20Example2(),
                "Example3": rb_settings.Gast20Example3()}
    N_low = 1000
    N_high = 10000
    if N_low < 100:
        N_step = 10
    elif N_low < 1000:
        N_step = 100
    elif N_low < 10000:
        N_step = 1000
    else:
        N_step = 10000
    Ns = np.arange(N_low, N_high, N_step)

    #Ns = np.arange(5000, 10000, 1000)
    #Ns = np.arange(500, 1000, 100)
    T = 1000
    act_frac = 0.4
    special = "mg" # maintaining good arms

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


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(precision=5, suppress=True)
    #search_for_WIP_counterexamples()
    #see_counterexamples()
    #Compare_policies_experiments()

    when_is_whittles_same_as_lp()


    ### To-do list:
    ### need to have a confidence interval by repeating multiple times
    ### add the curve of the naive randomized policy to the same plot
    ### rewrite the priority policy part with explicit code.
