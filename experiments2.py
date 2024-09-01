import numpy as np
import cvxpy as cp
import scipy
from discrete_RB import *
import rb_settings
import time
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import os


def run_policies(setting_name, policy_name, init_method, T, setting_path=None, note=None):
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
    Ns = list(range(100, 1100, 100))  #list(range(1500, 5500, 500)) # list(range(1000, 20000, 1000))
    for N in Ns:
        assert (N*act_frac).is_integer()
    num_reps = 1
    print()
    rb_settings.print_bandit(setting)

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
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

    reward_array = np.nan * np.empty((num_reps, len(Ns)))
    full_reward_trace = {}
    full_ideal_acts_trace = {}
    for i, N in enumerate(Ns):
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

            if policy_name == "id":
                policy = IDPolicy(setting.sspa_size, y, N, act_frac)
                for t in range(T):
                    cur_states = rb.get_states()
                    actions, num_ideal_acts = policy.get_actions(cur_states)
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
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
            elif policy_name == "twoset-v1":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U)
                print("eta=", policy.eta)
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(OL_set))
            elif policy_name == "twoset-integer":
                policy = TwoSetPolicy(setting.sspa_size, y, N, act_frac, U, rounding="misocp")
                print("eta=", policy.eta)
                for t in range(T):
                    cur_states = rb.get_states()
                    OL_set = policy.get_new_focus_set(cur_states=cur_states, last_OL_set=OL_set) ###
                    actions = policy.get_actions(cur_states, OL_set)
                    instant_reward = rb.step(actions)
                    total_reward += instant_reward
                    full_reward_trace[i,N].append(instant_reward)
                    full_ideal_acts_trace[i,N].append(len(OL_set))
            else:
                raise NotImplementedError
            avg_reward = total_reward / T
            reward_array[rep, i] = avg_reward
            avg_idea_frac = np.sum(np.array(full_ideal_acts_trace[i,N])) / (T*N)
            print("setting={}, policy={}, N={}, rep_id={}, avg reward = {}, avg ideal frac ={}, note={}".format(setting_name, policy_name, N, rep,
                                                                                   avg_reward, avg_idea_frac, note))

        if note is None:
            data_file_name = "fig_data/{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
        else:
            data_file_name = "fig_data/{}-{}-N{}-{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method, note)
        with open(data_file_name, 'wb') as f:
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
            pickle.dump(setting_and_data, f)


def figure_from_multiple_files():
    settings = ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"] #， "non-sa-big4"]     #
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-"]
    policy_markers = ["v",".","^","s","p","*"]
    policy_colors = ["m","c","y","r","g","b"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)"}
    setting2legend_position = {"random-size-10-dirichlet-0.05-(582)": (1,0.1), "random-size-10-dirichlet-0.05-(355)":(1,0.1),
                               "new2-eight-states":"center right", "three-states":(1,0.1), "non-sa":"lower right", "non-sa-big2":"center right"}
    setting2yrange = {"random-size-10-dirichlet-0.05-(582)": None, "random-size-10-dirichlet-0.05-(355)":None,
                      "new2-eight-states":None, "three-states":None, "non-sa":(0.7,1.025), "non-sa-big2":None}
    batch_means_dict = {}
    num_batches = 20
    Ns = np.array(list(range(100,1100,100))) #np.array(list(range(1500, 5500, 500))) # list(range(1000, 20000, 1000))
    init_method = "random"

    for setting_name in settings:
        for policy_name in policies:
            file_prefix = "{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], init_method)
            if setting_name in ["non-sa", "non-sa-big1", "non-sa-big2", "non-sa-big3", "non-sa-big4"] and (policy_name == "ftva"):
                file_prefix += "-T16e4"
            else:
                file_prefix += "-T2e4"
            file_names = [file_name for file_name in os.listdir("fig_data") if file_name.startswith(file_prefix)]
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
                with open("fig_data/"+file_name, 'rb') as f:
                    setting_and_data = pickle.load(f)
                    full_reward_trace = setting_and_data["full_reward_trace"]
                    for i,N in enumerate(Ns):
                        print("time horizon of {} = {}".format(file_name, len(full_reward_trace[i,N])))
                        cur_batch_size = int(len(full_reward_trace[i,N])/4)
                        # if policy_name == "ftva":
                        #     cur_batch_size = int(len(full_reward_trace[i,N])/10)
                        for t in range(0, len(full_reward_trace[i,N]), cur_batch_size):
                            batch_means_dict[(setting_name, policy_name)][i].append(np.mean(full_reward_trace[i, N][t:(t+cur_batch_size)]))
            for i in range(len(Ns)):
                assert len(batch_means_dict[(setting_name, policy_name)][i]) == num_batches
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
                # plt.plot(Ns, [0]*len(Ns), label=policy2label[policy_name],
                #      linewidth=1.5, linestyle=linestyle_str[i])
                pass
            else:
                plt.errorbar(Ns, np.mean(batch_means_dict[(setting_name, policy_name)], axis=1) / upper_bound,
                             yerr=2*np.std(batch_means_dict[(setting_name, policy_name)], axis=1)/np.sqrt(num_batches) / upper_bound,
                             label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                             marker=policy_markers[i], markersize=8, color=policy_colors[i])
            plt.xlabel("N", fontsize=14)
        # plt.title("Simulations for {} example".format(setting_name))
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

    # for setting_name in ["non-sa-big4"]: #["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
    #     for policy_name in ["setexp-priority", "whittle"]:  # ["id", "setexp", "setopt", "ftva", "lppriority", "setexp-priority", "twoset-v1"]
    #         for rep_id in range(1,6):
    #             tic = time.time()
    #             run_policies(setting_name, policy_name, "random", 20000, note="T2e4r{}".format(rep_id))
    #             toc = time.time()
    #             print("when T=20000, total time per policy =", toc-tic)

    ## random three-state examples
    # for i in range(5):
    #     setting_name = "random-size-3-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["twoset-v1"]:
    #         run_policies(setting_name, policy_name, "random", 10000, setting_path)

    ## random four-state examples
    # for i in range(3):
    #     setting_name = "random-size-4-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["setexp-priority"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         run_policies(setting_name, policy_name, "random", 10000, setting_path)

    ## random 8-state examples
    # for laziness in [0.1, 0.2, 0.3]:
    #     setting_name = "random-size-8-uniform-nzR0-lazy-{}-({})".format(laziness,0)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["id", "setexp"]: #, "setopt", "ftva", "lppriority", "setopt-priority", "whittle"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         run_policies(setting_name, policy_name, "random", 10000, setting_path)


    # ## random 10-state dirichlet examples
    # for i in [582]: #[137, 355, 582]:
    #     setting_name = "random-size-10-dirichlet-0.05-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["id", "setexp", "ftva", "lppriority", "setexp-priority", "whittle"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         for rep_id in range(2, 6):
    #             run_policies(setting_name, policy_name, "random", 20000, setting_path, note="T2e4r{}".format(rep_id))



    # ## random 16-state RG examples
    # for i in range(3):
    #     setting_name = "RG-16-square-uniform-thresh=0.5-uniform-({})".format(i)
    #     setting_path = "setting_data/" + setting_name
    #     setting = rb_settings.ExampleFromFile(setting_path)
    #     for policy_name in ["id", "setexp"]: #, "setopt", "ftva", "lppriority", "setopt-priority", "whittle"]: #["id", "setexp", "setopt", "ftva", "lppriority", "setopt-priority", "twoset-v1"]:
    #         run_policies(setting_name, policy_name, "random", 10000, setting_path)

    # figure_from_multiple_files()

    setting_name = "random-size-10-dirichlet-0.05-(355)"
    setting_path = "setting_data/" + setting_name
    setting = rb_settings.ExampleFromFile(setting_path)
    rb_settings.print_bandit(setting, latex_format=True)
