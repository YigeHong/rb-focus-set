import rb_settings
from discrete_RB import *
import os

def test_local_stability(setting, priority_list, act_frac, try_steps, eps, verbose=False):
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


def search_for_counterexamples():
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
        has_cycle = test_local_stability(setting, priority_list, act_frac, try_steps=try_steps, eps=eps)
        if has_cycle:
            num_indexable_cycles += 1
            print("cycle found.")
            rb_settings.save_bandit(setting, f_path, other_params={"act_frac": act_frac})
            rb_settings.print_bandit(setting)

    print("fraction of nonindexable:", num_maybe_nonindexable / num_of_settings)
    print("fraction of indexable cycle", num_indexable_cycles / num_of_settings)


def visualize_counterexamples():
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
        has_cycle = test_local_stability(setting, priority_list, act_frac, try_steps=1000, eps=1e-4, verbose=True)
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
