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

def figure_avg_over_settings(note=None):
    """
    Plotting function that reads data files with Ns to combine into one plot.
    """
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["new2-eight-states-045"]+["conveyor-belt-nd-12"]+["random-size-8-uniform-({})".format(i) for i in [1, 6, 0]]
    # other possible settings ["random-size-8-uniform-({})".format(i) for i in [1]] # ["mix-random-size-10-dirichlet-0.05-({})-(2270)-ratio-0.95".format(i) for i in [1436, 6265]] #["stable-size-10-dirichlet-0.05-({})".format(i) for i in [4339]]#, 4149, 4116, 2667, 2270, 9632]]
    settings = ["random-size-8-uniform-({})".format(i) for i in range(8)]
    setting_batch_name = "randomto8"
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
    batch_size_mode = "adaptive" #"fixed" or "adaptive"
    batch_size = 8000 # only if batch_size_mode = "fixed"
    burn_in_batch = 1
    N_range = [200, 10000] # closed interval
    agg_modes = ["avg", "median", "cdf"] # avg, median, cdf
    only_plot_N = 10000
    init_method = "random"
    file_dirs = ["fig_data_server_0922", "fig_data_server_0925"]
    N_tick_ratio = 1000
    tick_locs =  np.arange(0, 11000, 2000)
    tick_labels = [int(N/N_tick_ratio) for N in tick_locs]
    assert np.allclose(np.array(tick_labels)*N_tick_ratio, np.array(tick_locs)), print(tick_labels, tick_locs)
    legend_on = False

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

    Ns_common = []  # different setting and policy could have different N; Ns_union records N that appears in all
    opt_gap_ratios_across_settings = {}   # dict of (policy, reward_array)

    for setting_id, setting_name in enumerate(settings):
        upper_bound = all_batch_means[(setting_name, "upper bound")]
        if setting_name in setting2truncate:
            truncate_level = setting2truncate[setting_name]
        else:
            truncate_level = truncate_level_default
        # if mode == "opt-ratio":
        #     plt.plot([N_range[0], N_range[1]], np.array([1, 1]), label="Upper bound", linestyle="--", color="k")
        for policy_name in policies:
            if (policy_name == "whittle") and ((setting_name, policy_name) not in all_batch_means):
                continue
            if policy_name in skip_policies:
                continue
            else:
                Ns_cur = []
                avg_rewards_cur = []
                # yerrs_cur = []
                cur_policy_batch_means = all_batch_means[(setting_name, policy_name)]
                for N in cur_policy_batch_means:
                    Ns_cur.append(N)
                    avg_rewards_cur.append(np.mean(cur_policy_batch_means[N]))
                    # yerrs_cur.append(1.96 * np.std(cur_policy_batch_means[N]) / np.sqrt(len(cur_policy_batch_means[N])))
                Ns_cur = np.array(Ns_cur)
                avg_rewards_cur = np.array(avg_rewards_cur)
                # yerrs_cur = np.array(yerrs_cur)
                sorted_indices = np.argsort(Ns_cur)
                Ns_cur_sorted = Ns_cur[sorted_indices]
                avg_rewards_cur_sorted = avg_rewards_cur[sorted_indices]
                # yerrs_cur_sorted = yerrs_cur[sorted_indices]
                print(setting_name, policy_name, avg_rewards_cur_sorted)

                if len(Ns_common) == 0: # the first (setting, policy) pair
                    Ns_common = Ns_cur_sorted
                if policy_name not in opt_gap_ratios_across_settings:
                    opt_gap_ratios_across_settings[policy_name] = np.zeros((len(settings), len(Ns_common)))
                for N_ind, N in enumerate(Ns_common):
                    if N not in Ns_cur_sorted:
                        opt_gap_ratios_across_settings[policy_name][setting_id,N_ind] = np.nan # no data, set to nan
                    else:
                        opt_gap_ratios_across_settings[policy_name][setting_id,N_ind] += 1 - avg_rewards_cur_sorted[N_ind] / upper_bound

    # pop N that lacks data for some (setting, policy) pair
    ind_to_pop = []
    for N_ind, N in enumerate(Ns_common):
        for policy_name in policies:
            if policy_name in skip_policies:
                continue
            if np.isnan(opt_gap_ratios_across_settings[policy_name][:,N_ind]).any():
                ind_to_pop.append(N_ind)
    for N_ind in sorted(ind_to_pop, reverse=True):
        Ns_common.pop(N_ind)
        for policy_name in policies:
            if policy_name in skip_policies:
                continue
            np.delete(opt_gap_ratios_across_settings[policy_name], N_ind, axis=1)

    for agg_mode in agg_modes:
        Ns_common = np.array(Ns_common)
        max_value_for_ylim = 0
        for i, policy_name in enumerate(policies):
            if policy_name in skip_policies:
                continue
            if agg_mode == "cdf":
                N_ind = np.nonzero(Ns_common==only_plot_N)[0][-1]
                # print("log", np.nonzero(Ns_common==only_plot_N))
                # print(opt_gap_ratios_across_settings[policy_name][:,N_ind])
                # print(np.sort(opt_gap_ratios_across_settings[policy_name][:,N_ind]))
                # print(sorted(opt_gap_ratios_across_settings[policy_name][:,N_ind]))
                cdf_jump_pts = sorted(opt_gap_ratios_across_settings[policy_name][:,N_ind]) * only_plot_N
                plt.plot(cdf_jump_pts, np.linspace(0,1,len(cdf_jump_pts)),label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                            marker=policy_markers[i], markersize=8, color=policy_colors[i])
            else:
                if agg_mode == "avg":
                    opt_gap_ratio_cur = np.average(opt_gap_ratios_across_settings[policy_name], axis=0)
                elif agg_mode == "median":
                    opt_gap_ratio_cur = np.median(opt_gap_ratios_across_settings[policy_name], axis=0)
                else:
                    raise NotImplementedError
                plt.plot(Ns_common, opt_gap_ratio_cur * Ns_common,label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                            marker=policy_markers[i], markersize=8, color=policy_colors[i])
                max_value_for_ylim = max(max_value_for_ylim, np.max(opt_gap_ratio_cur * Ns_common))

        if agg_mode == "cdf":
            plt.xlabel("Total optimality gap ratio", fontsize=20)
            plt.ylabel("CDF", fontsize=20)
        else:
            plt.xlabel("N (x{})".format(N_tick_ratio), fontsize=20)
            plt.xticks(ticks=tick_locs, labels=tick_labels, fontsize=20)
            plt.ylabel("Total optimality gap ratio", fontsize=20)
            plt.ylim((-0.06*max_value_for_ylim, max_value_for_ylim*1.05))
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.grid()
        if legend_on:
            plt.legend(fontsize=20)
        agg_mode_with_param = agg_mode + str(only_plot_N) if agg_mode == "cdf" else agg_mode
        plt.savefig("figs3/total-gap-ratio-{}-{}-N{}-{}-{}.png".format(setting_batch_name, agg_mode_with_param, N_range[0], N_range[1], init_method))
        # plt.savefig("formal_figs_exponential/total-gap-ratio-{}.png".format(setting_batch_name))
        plt.show()


def figure_from_multiple_files_flexible_N(note=None):
    """
    Plotting function that reads data files with Ns to combine into one plot.
    """
    # setting names in the unichain aperiodicity paper: ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]
    # setting names in the exponential paper ["new2-eight-states-045"]+["conveyor-belt-nd-12"]+["random-size-8-uniform-({})".format(i) for i in [1, 6, 0]]
    # other possible settings ["random-size-8-uniform-({})".format(i) for i in [1]] # ["mix-random-size-10-dirichlet-0.05-({})-(2270)-ratio-0.95".format(i) for i in [1436, 6265]] #["stable-size-10-dirichlet-0.05-({})".format(i) for i in [4339]]#, 4149, 4116, 2667, 2270, 9632]]
    settings = ["random-size-8-uniform-({})".format(i) for i in [1]]
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
    N_range = [200, 10000] # closed interval
    init_method = "random"
    mode = "total-opt-gap-ratio" # "opt-ratio" or "total-opt-gap-ratio" or "log-opt-gap-ratio"
    file_dirs = ["fig_data", "fig_data_server_0922", "fig_data_server_0925", "fig_data_server_0928", "fig_data_server_1001"]
    N_tick_ratio = 1000
    tick_locs =  np.arange(0, 11000, 2000)
    tick_labels = [int(N/N_tick_ratio) for N in tick_locs]
    assert np.allclose(np.array(tick_labels)*N_tick_ratio, np.array(tick_locs)), print(tick_labels, tick_locs)
    legend_on = False


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
                Ns_cur = []
                avg_rewards_cur = []
                yerrs_cur = []
                cur_policy_batch_means = all_batch_means[(setting_name, policy_name)]
                for N in cur_policy_batch_means:
                    Ns_cur.append(N)
                    avg_rewards_cur.append(np.mean(cur_policy_batch_means[N]))
                    yerrs_cur.append(1.96 * np.std(cur_policy_batch_means[N]) / np.sqrt(len(cur_policy_batch_means[N])))
                Ns_cur = np.array(Ns_cur)
                avg_rewards_cur = np.array(avg_rewards_cur)
                yerrs_cur = np.array(yerrs_cur)
                sorted_indices = np.argsort(Ns_cur)
                Ns_cur_sorted = Ns_cur[sorted_indices]
                avg_rewards_cur_sorted = avg_rewards_cur[sorted_indices]
                yerrs_cur_sorted = yerrs_cur[sorted_indices]
                print(setting_name, policy_name, avg_rewards_cur_sorted, yerrs_cur_sorted)
                ## special handling of "random-size-8-uniform-(1)", manually filter out bad data for high precision:...
                # if (setting_name == "random-size-8-uniform-(1)"):
                #     if policy_name in ["whittle", "lppriority"]:
                #         show_until = bisect.bisect_left(Ns_cur_sorted, 6000)
                #         Ns_cur_sorted = Ns_cur_sorted[0:show_until]
                #         avg_rewards_cur_sorted = avg_rewards_cur_sorted[0:show_until]
                #         yerrs_cur_sorted = yerrs_cur_sorted[0:show_until]
                    # for j, N in enumerate(Ns_cur_sorted):
                    #     if policy_name in ["whittle", "lppriority"] and (N>=6000):
                            # avg_rewards_cur_sorted[j] = upper_bound
                            # yerrs_cur_sorted[j] = 0
                        # elif (policy_name == "twoset-faithful") and (N > 8000):
                        #     avg_rewards_cur_sorted[j] = upper_bound
                        #     yerrs_cur_sorted[j] = 0

                if not plot_CI:
                    if mode == "opt-ratio":
                        cur_curve = plt.plot(Ns_cur_sorted, avg_rewards_cur_sorted / upper_bound)
                    elif mode == "total-opt-gap-ratio":
                        cur_curve = plt.plot(Ns_cur_sorted, (upper_bound - avg_rewards_cur_sorted) * Ns_cur_sorted / upper_bound)
                        if policy_name not in ["lppriority", "whittle"]:
                            max_value_for_ylim = max(max_value_for_ylim, np.max((upper_bound - avg_rewards_cur_sorted) * Ns_cur_sorted / upper_bound))
                    elif mode == "log-opt-gap-ratio":
                        cur_curve = plt.plot(Ns_cur_sorted, np.log10((upper_bound - avg_rewards_cur_sorted)/upper_bound))
                    else:
                        raise NotImplementedError
                    cur_curve.set(label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                marker=policy_markers[i], markersize=8, color=policy_colors[i])
                else:
                    if mode == "opt-ratio":
                        plt.errorbar(Ns_cur_sorted, avg_rewards_cur_sorted / upper_bound,
                                     yerr=yerrs_cur_sorted / upper_bound, label=policy2label[policy_name],
                                     linewidth=1.5, linestyle=linestyle_str[i], marker=policy_markers[i], markersize=8,
                                     color=policy_colors[i])
                    elif mode == "total-opt-gap-ratio":
                        Ns_include_zero = np.insert(Ns_cur_sorted, 0, 0)
                        ys = (upper_bound - avg_rewards_cur_sorted) * Ns_cur_sorted / upper_bound
                        ys_include_zero = np.insert(ys, 0, 0)
                        yerrs_include_zero = np.insert(yerrs_cur_sorted * Ns_cur_sorted / upper_bound, 0, 0)
                        plt.errorbar(Ns_include_zero, ys_include_zero,
                                     yerr=yerrs_include_zero, label=policy2label[policy_name],
                                     linewidth=1.5, linestyle=linestyle_str[i], marker=policy_markers[i], markersize=8,
                                     color=policy_colors[i])
                        if policy_name not in ["lppriority", "whittle"]:
                            max_value_for_ylim = max(max_value_for_ylim, np.max((upper_bound - avg_rewards_cur_sorted+yerrs_cur_sorted) * Ns_cur_sorted / upper_bound))
                    elif mode == "log-opt-gap-ratio":
                        upper_CI_truncated = np.clip((upper_bound - avg_rewards_cur_sorted + yerrs_cur_sorted) / upper_bound, truncate_level, None)
                        lower_CI_truncated = np.clip((upper_bound - avg_rewards_cur_sorted - yerrs_cur_sorted) / upper_bound, truncate_level, None)
                        mean_truncated = np.clip((upper_bound - avg_rewards_cur_sorted) / upper_bound, truncate_level, None)
                        print(upper_CI_truncated, lower_CI_truncated)
                        plt.errorbar(Ns_cur_sorted, np.log10(mean_truncated),
                                 yerr=np.stack([np.log10(mean_truncated) - np.log10(lower_CI_truncated), np.log10(upper_CI_truncated) - np.log10(mean_truncated)]),
                                 label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                                 marker=policy_markers[i], markersize=8, color=policy_colors[i])
                    else:
                        raise NotImplementedError
                # print("{} {}, yerrs/upper_bound = {}".format(setting_name, policy_name, yerrs_cur_sorted / upper_bound))
                # print("{} {}, relative error = {}".format(setting_name, policy_name, yerrs_cur_sorted / np.clip(upper_bound - avg_rewards_cur_sorted, 0, None) / np.log(10)))
            plt.xlabel("N (x{})".format(N_tick_ratio), fontsize=20)
        plt.xticks(ticks=tick_locs, labels=tick_labels, fontsize=20)
        if mode == "opt-ratio":
            plt.ylabel("Optimality ratio", fontsize=20)
        elif mode == "total-opt-gap-ratio":
            plt.ylabel("Total optimality gap ratio", fontsize=20)
        elif mode == "log-opt-gap-ratio":
            plt.ylabel("Log optimality gap ratio", fontsize=20)
        else:
            raise NotImplementedError
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.grid()
        if (setting_name == "random-size-8-uniform-(1)") and (mode == "log-opt-gap-ratio"):
            pass # plt.legend(fontsize=14, loc="center right")
        elif not legend_on:
            pass
        else:
            plt.legend(fontsize=20)
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
    figure_avg_over_settings()
