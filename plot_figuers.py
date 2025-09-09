#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates the plots and figures from saved experimental data of experiments2.py

This script reproduces the figures for the paper "Unichain and Aperiodicity
are Sufficient for Asymptotic Optimality of Restless Bandits" (hereafter
referred to as "RB-unichain").

Note: only figure_from_multiple_files is relevant to the RB-unichain paper.
"""

__author__ = "Yige Hong"
__date__ = "2025-09-06"
__version__ = "1.0"


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


def filter_settings(setting_names, setting_dir, have_local_unstable):
    if have_local_unstable:
        return setting_names
    else:
        new_setting_name_list = []
        for setting_name in setting_names:
            setting_path = os.path.join(setting_dir, setting_name)
            setting = rb_settings.ExampleFromFile(setting_path)
            analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
            y = analyzer.solve_lp()[1]
            print("min(y(sneu,1), y(sneu,0)) = ", y2nondegeneracy(y))
            W = analyzer.compute_W(abstol=1e-10)[0]
            print("2*lambda_W = ", 2*np.linalg.norm(W, ord=2))
            if setting_name == "eight-states":
                priority_list = analyzer.solve_LP_Priority(fixed_dual=0)
            else:
                priority_list = analyzer.solve_LP_Priority()
            print("priority list =", priority_list)
            whittle_priority = analyzer.solve_whittles_policy()
            print("Whittle priority=", whittle_priority)
            U = analyzer.compute_U(abstol=1e-10)[0]
            if U is not np.inf:
                new_setting_name_list.append(setting_name)
        return new_setting_name_list



def figure_from_multiple_files():
    """
    Generate Figures for comparing different policies' optimality gap ratios as N increases
    Requires first running `experiments2.py`, and then read the saved data from folder `fig_data`
    The generated figures will be saved in the folder `figs2`
    """
    settings = ["three-states", "new2-eight-states", "non-sa", "non-sa-big2"] \
               + ["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]]
    policies = ["whittle", "lppriority", "ftva", "setexp", "setexp-priority", "id"]
    linestyle_str = ["-.", "-", "--", "-.", "--", "-"]
    policy_markers = ["v",".","^","s","p","*"]
    policy_colors = ["m","c","y","r","g","b"]
    policy2label = {"id":"ID policy", "setexp":"Set expansion", "lppriority":"LP index policy",
                    "whittle":"Whittle index policy", "ftva":"FTVA", "setexp-priority":"Set expansion (with LP index)",
                    "twoset-v1":"Two set v1"}
    # skip showing some policies without changing the color and markers of the rest;
    # should be [] for normal runs
    skip_policies = []
    # Some ad-hoc code for legend positions and yranges
    setting2legend_position = {"random-size-10-dirichlet-0.05-(582)": (1,0.1), "random-size-10-dirichlet-0.05-(355)":(1,0.1),
                               "new2-eight-states":"center right", "three-states":(1,0.1), "non-sa":"lower right", "non-sa-big2":"center right"}
    setting2yrange = {"random-size-10-dirichlet-0.05-(582)": None, "random-size-10-dirichlet-0.05-(355)":None,
                      "new2-eight-states":None, "three-states":None, "non-sa":(0.7,1.025), "non-sa-big2":None}
    # The ranges of N to be plotted
    Ns = np.array(list(range(100,1100,100))) #np.array(list(range(1500, 5500, 500))) # list(range(1000, 20000, 1000))
    main_init_method = "random"
    file_dir = "fig_data"

    setting_policy_init_tuples = [(s_name, p_name, main_init_method) for s_name in settings for p_name in policies]
    # add some curves manually, commented out for normal runs
    # setting_policy_init_tuples.append(("non-sa", "ftva", "bad"))
    # setting_policy_init_tuples.append(("non-sa", "ftva", "given"))
    # setting_policy_init_tuples.append(("non-sa-big2", "ftva", "bad"))
    # setting_policy_init_tuples.append(("non-sa-big2", "ftva", "given"))

    batch_means_dict = {}
    for setting_name, policy_name, cur_init_method in setting_policy_init_tuples:
        file_prefix = "{}-{}-N{}-{}-{}".format(setting_name, policy_name, Ns[0], Ns[-1], cur_init_method)
        print(file_prefix)

        # Ad-hoc code for only using data that runs for 16e4 time steps for FTVA
        if cur_init_method not in ["bad", "given"]:
            if setting_name in ["non-sa", "non-sa-big1", "non-sa-big2", "non-sa-big3", "non-sa-big4"] and (policy_name == "ftva"):
                file_prefix += "-T16e4"
            else:
                file_prefix += "-T2e4"
        # find all data file names with matching prefixes
        file_names = [file_name for file_name in os.listdir(file_dir) if file_name.startswith(file_prefix)]
        print("{}:{}".format(file_prefix, file_names))
        # raise error if no file is found for the given RB instance and policy,
        # unless the policy is Whittle index and the RB instance is non-indexable
        if len(file_names) == 0:
            if policy_name == "whittle":
                continue
            else:
                raise FileNotFoundError("no file with prefix {}".format(file_prefix))
        # calculate batch means from the files
        batch_means_dict[(setting_name, policy_name, cur_init_method)] = [[] for i in range(len(Ns))] # shape len(Ns)*num_batches
        for file_name in file_names:
            with open(os.path.join(file_dir, file_name), 'rb') as f:
                setting_and_data = pickle.load(f)
                full_reward_trace = setting_and_data["full_reward_trace"]
                for i,N in enumerate(Ns):
                    print("time horizon of {} = {}".format(file_name, len(full_reward_trace[i,N])))
                    cur_batch_size = int(len(full_reward_trace[i,N])/4)
                    for t in range(0, len(full_reward_trace[i,N]), cur_batch_size):
                        batch_means_dict[(setting_name, policy_name, cur_init_method)][i].append(np.mean(full_reward_trace[i, N][t:(t+cur_batch_size)]))
        batch_means_dict[(setting_name, policy_name, cur_init_method)] = np.array(batch_means_dict[(setting_name, policy_name, cur_init_method)])
        print(setting_name, policy_name, np.mean(batch_means_dict[(setting_name, policy_name, cur_init_method)], axis=1), np.std(batch_means_dict[(setting_name, policy_name, cur_init_method)], axis=1))


    for setting_name in settings:
        # some earlier simulation data files do not include upper bounds, so we add them manually
        if setting_name == "eight-states":
            upper_bound = 0.0125
        elif setting_name == "three-states":
            upper_bound = 0.12380016733626052
        elif setting_name == "non-sa":
            upper_bound = 1
        else:
            files_w_prefix = [filename for filename in os.listdir("fig_data")
                              if filename.startswith("{}-{}-N{}-{}-{}".format(setting_name, "ftva", 100, 1000, main_init_method))]
            with open("fig_data/"+files_w_prefix[0], 'rb') as f:
                setting_and_data = pickle.load(f)
                upper_bound = setting_and_data["upper bound"]

        ## -----Plot the upper bound and the policies-----
        plt.plot(Ns, np.array([1] * len(Ns)), label="Upper bound", linestyle="--", color="k")
        for i, policy_name in enumerate(policies):
            if (policy_name == "whittle") and ((setting_name, policy_name) not in batch_means_dict):
                pass
            elif policy_name in skip_policies:
                pass
            else:
                cur_batch_means = batch_means_dict[(setting_name, policy_name, main_init_method)]
                plt.errorbar(Ns, np.mean(cur_batch_means, axis=1) / upper_bound,
                             yerr=2*np.std(cur_batch_means, axis=1)/np.sqrt(len(cur_batch_means)) / upper_bound,
                             label=policy2label[policy_name], linewidth=1.5, linestyle=linestyle_str[i],
                             marker=policy_markers[i], markersize=8, color=policy_colors[i])

        # manually add some curves, commented out under normal runs
        # for additional_init in ["bad", "given"]:
        #     cur_batch_means = batch_means_dict[(setting_name, "ftva", additional_init)]
        #     temp_linestyle = "-" if additional_init == "bad" else "-."
        #     temp_marker = "o" if additional_init == "bad" else "^"
        #     temp_label = "FTVA (init at 0)" if additional_init == "bad" else "FTVA (fixed random init)"
        #     plt.errorbar(Ns, np.mean(cur_batch_means, axis=1) / upper_bound,
        #                          yerr=2*np.std(cur_batch_means, axis=1)/np.sqrt(len(cur_batch_means)) / upper_bound,
        #                          label=temp_label, linestyle=temp_linestyle,
        #                           marker=temp_marker, linewidth=1.5, markersize=8)

        ## -----Figure settings-----
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
        plt.savefig("figs2/{}-N{}-{}-{}.pdf".format(setting_name, Ns[0], Ns[-1], main_init_method))
        plt.show()

if __name__ == "__main__":
    # Run this function to plot figures;
    # Modify the code inside the function to change the printing options
    figure_from_multiple_files()
