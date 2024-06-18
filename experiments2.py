import numpy as np
import cvxpy as cp
import scipy
from discrete_RB import *
import rb_settings
import time
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl


def run_policies(setting_name, policy_name, init_method, T, setting_path=None):
    if setting_name == "eight-states":
        probs_L, probs_R, action_script, suggest_act_frac = rb_settings.ConveyorExample.get_parameters(
            "eg4action-gap-tb", 8)
        setting = rb_settings.ConveyorExample(8, probs_L, probs_R, action_script, suggest_act_frac)
    elif setting_name == "three-states":
        setting = rb_settings.Gast20Example2()
    elif setting_name == "non-sa":
        setting = rb_settings.NonSAExample()
    elif setting_path is not None:
        setting = rb_settings.ExampleFromFile(setting_path)
    else:
        raise NotImplementedError
    act_frac = setting.suggest_act_frac
    Ns = list(range(100, 1100, 100))
    num_reps = 1

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    W = analyzer.compute_W(abstol=1e-10)[0]
    if setting_name == "eight-states":
        priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
    else:
        priority_list = analyzer.solve_LP_Priority()
    # print(W)
    # eigs = np.linalg.eigh(W)
    # for eig in eigs:
    #     print(np.sort(np.abs(eig)))
    # cpi = np.array([1,1,1,1,0,0,0,0])
    # print(np.linalg.inv(W))
    # print(np.sqrt(np.matmul(np.matmul(cpi.T, np.linalg.inv(W)), cpi)))
    # exit()

    reward_array = np.nan * np.empty((num_reps, len(Ns)))
    for i, N in enumerate(Ns):
        for rep in range(num_reps):
            if init_method == "random":
                init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
            elif init_method == "same":
                init_states = np.zeros((N,))
            else:
                raise NotImplementedError
            rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)
            total_reward = 0
            conformity_count = 0
            non_shrink_count = 0
            focus_set = np.array([], dtype=int)

            if policy_name == "id":
                policy = IDPolicy(setting.sspa_size, y, N, act_frac)
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
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
                    conformity_count += conformity_flag
                    non_shrink_count += non_shrink_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
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
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
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
                    conformity_count += conformity_flag
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
            elif policy_name == "ftva":
                policy = FTVAPolicy(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, y=y, N=N,
                                    act_frac=act_frac, init_virtual=None)
                for t in range(T):
                    prev_state = rb.get_states()
                    actions, virtual_actions = policy.get_actions(prev_state)
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    new_state = rb.get_states()
                    policy.virtual_step(prev_state, new_state, actions, virtual_actions)
            elif policy_name == "lppriority":
                policy = PriorityPolicy(setting.sspa_size, priority_list, N=N, act_frac=act_frac)
                for t in range(T):
                    cur_states = rb.get_states()
                    actions = policy.get_actions(cur_states)
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
            else:
                raise NotImplementedError
            avg_reward = total_reward / T
            reward_array[rep, i] = avg_reward
            print("setting={}, policy={}, N={}, rep_id={}, avg reward = {}".format(setting_name, policy_name, N, rep,
                                                                                   avg_reward))

        with open("fig_data/{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method), 'wb') as f:
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
                "y": y,
                "W": W,
                "upper bound": analyzer.opt_value
            }
            pickle.dump(setting_and_data, f)


def figure_from_multiple_files():
    settings = ["eight-states", "three-states", "non-sa"] + ["random-size-3-uniform-({})".format(i) for i in range(5)]  # ["eight-states", "three-states", "non-sa"]
    policies = ["id", "ftva", "lppriority", "setexp", "setopt"]  # ["id", "setexp", "setopt", "ftva", "lppriority"]
    reward_array_dict = {}
    Ns = np.array(list(range(100, 1100, 100)))
    init_method = "random"

    for setting_name in settings:
        for policy_name in policies:
            if (policy_name in ["id", "setexp", "setopt"]) or \
                    (setting_name not in ["eight-states", "three-states"]):
                with open("fig_data/{}-{}-N{}-{}-{}".format(setting_name, policy_name, 100, 1000, init_method), 'rb') as f:
                    setting_and_data = pickle.load(f)
                    reward_array_dict[(setting_name, policy_name)] = setting_and_data["reward_array"]
                    print(setting_name, policy_name, reward_array_dict[(setting_name, policy_name)])

    temp_name_suffix = "bad" if init_method == "same" else "random"
    with open("fig_data/Formal-{}-N{}-{}-{}".format("Figure2Example", 100, 1100, temp_name_suffix), 'rb') as f:
        setting_and_data = pickle.load(f)
        reward_array_dict[("eight-states", "ftva")] = setting_and_data["sp_avg_rewards"]
    with open("fig_data/Formal-{}-N{}-{}-{}".format("Figure2Example", 100, 1100, temp_name_suffix), 'rb') as f:
        setting_and_data = pickle.load(f)
        reward_array_dict[("eight-states", "lppriority")] = setting_and_data["lp_avg_rewards"]

    ### warning: initialization method of this simulation is different; but I guess that do not affect the result
    with open("fig_data/Formal-{}-N{}-{}-{}".format("Figure1Example", 100, 1100, "default"), 'rb') as f:
        setting_and_data = pickle.load(f)
        reward_array_dict[("three-states", "ftva")] = setting_and_data["sp_avg_rewards"]
    with open("fig_data/Formal-{}-N{}-{}-{}".format("Figure1Example", 100, 1100, "default"), 'rb') as f:
        setting_and_data = pickle.load(f)
        reward_array_dict[("three-states", "lppriority")] = setting_and_data["wip_avg_rewards"]

    for setting_name in settings:
        if setting_name == "eight-states":
            upper_bound = 0.0125
        elif setting_name == "three-states":
            upper_bound = 0.12380016733626052
        elif setting_name == "non-sa":
            upper_bound = 1
        else:
            with open("fig_data/{}-{}-N{}-{}-{}".format(setting_name, "id", 100, 1000, init_method), 'rb') as f:
                setting_and_data = pickle.load(f)
                upper_bound = setting_and_data["upper bound"]
        plt.plot(Ns, np.array([upper_bound] * len(Ns)), label="upper bound", linestyle="--")
        for policy_name in policies:
            plt.plot(Ns, np.average(reward_array_dict[(setting_name, policy_name)], axis=0), label=policy_name,
                     linewidth=1.5)
            plt.xlabel("N", fontsize=10)
        plt.title("Simulations for {} example".format(setting_name))
        plt.xticks(fontsize=10)
        plt.ylabel("Avg Reward", fontsize=10)
        plt.yticks(fontsize=10)
        # plt.ylim([0, 1.1*upper_bound]) ###
        plt.grid()
        plt.legend()
        plt.savefig("figs2/{}-N{}-{}-{}".format(setting_name, Ns[0], Ns[-1], init_method))
        plt.show()


if __name__ == "__main__":
    for setting_name in ["eight-states", "three-states", "non-sa"]:   #["eight-states", "three-states", "non-sa"]:
        for policy_name in ["setopt-tight"]:
            run_policies(setting_name, policy_name, "random", 10000)
    #
    # # ## random examples
    # for i in range(5):
    #     setting_path = "setting_data/random-size-3-uniform-({})".format(i)
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     rb_settings.print_bandit(setting)
    #     setting_name = "random-size-3-uniform-({})".format(i)
    #     for policy_name in ["setexp", "setopt"]:
    #         run_policies(setting_name, policy_name, "random", 10000, setting_path)

    # figure_from_multiple_files()

