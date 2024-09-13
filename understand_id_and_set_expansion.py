import numpy as np
import cvxpy as cp
import scipy
import pickle
from time import time
from matplotlib import pyplot as plt
from discrete_RB import *
import rb_settings



def convert_name_to_setting(setting_name, setting_path):
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
    return setting


def visualize_ideal_action_persistency(setting_name):
    setting_path = "setting_data/" + setting_name
    setting = convert_name_to_setting(setting_name, setting_path)

    # setting = rb_settings.ExampleFromFile("setting_data/random-size-3-uniform-(0)")
    N = 500
    act_frac = setting.suggest_act_frac
    rb_settings.print_bandit(setting)

    T = 1000
    T_ahead = 200
    burn_in = 5000
    init_method = "random" # "random" or "bad

    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, act_frac)
    y = analyzer.solve_lp()[1]
    LP_index_priority = analyzer.solve_LP_Priority()
    W = analyzer.compute_W(abstol=1e-10)[0]
    print("{}, 2*lambda_W = {}".format(setting_name, 2*np.linalg.norm(W, ord=2)))

    if init_method == "random":
        init_states = np.random.choice(np.arange(0, setting.sspa_size), N, replace=True)
    elif init_method == "bad":
        init_states = np.random.choice(np.arange(4, setting.sspa_size), N, replace=True)
    else:
        raise NotImplementedError
    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)

    id_policy = IDPolicy(setting.sspa_size, y, N, act_frac)
    setexp_policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)

    ID_ideal_act_agreement_maps = []
    for t in range(0, burn_in+T+T_ahead):
        cur_states = rb.get_states()
        actions, Npibs, ideal_actions = id_policy.get_actions(cur_states, output_ideal_actions=True)
        rb.step(actions)
        if t >= burn_in:
            ID_ideal_act_agreement_maps.append(actions==ideal_actions)
    ID_ideal_act_agreement_maps = np.array(ID_ideal_act_agreement_maps)

    ID_persistency_nums = []
    for t in range(T):
        cur_persistency_map = np.product(ID_ideal_act_agreement_maps[t:(t+T_ahead),:], axis=0) #test this part
        ID_persistency_nums.append(np.sum(cur_persistency_map))

    rb = RB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, N, init_states)

    setexp_ideal_act_agreement_maps = []
    focus_set = np.array([], dtype=int)
    for t in range(0, burn_in+T+T_ahead):
        cur_states = rb.get_states()
        focus_set, non_shrink_flag = setexp_policy.get_new_focus_set(cur_states=cur_states,
                                                                     last_focus_set=focus_set)
        actions, _, ideal_actions = setexp_policy.get_actions(cur_states, focus_set, output_ideal_actions=True)
        # actions, _, ideal_actions = setexp_policy.get_actions(cur_states, focus_set, tb_rule="priority",
        #                                                       tb_priority=LP_index_priority, output_ideal_actions=True)
        rb.step(actions)
        if t > burn_in:
            setexp_ideal_act_agreement_maps.append(actions==ideal_actions)
    setexp_ideal_act_agreement_maps = np.array(setexp_ideal_act_agreement_maps)

    setexp_persistency_nums = []
    for t in range(T):
        cur_persistency_map = np.product(setexp_ideal_act_agreement_maps[t:(t+T_ahead),:], axis=0) #test this part
        setexp_persistency_nums.append(np.sum(cur_persistency_map))

    fig = plt.figure()
    plt.plot(np.arange(burn_in, burn_in+T), np.array(ID_persistency_nums) / N, label="ID policy")
    plt.plot(np.arange(burn_in, burn_in+T), np.array(setexp_persistency_nums) / N, label="Set-expansion policy")
    plt.xlabel("Time step", fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Fraction of arms", fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim([0,1])
    plt.legend(fontsize=14)#, loc="center right")
    plt.tight_layout()
    plt.savefig("figs2/persistency-{}-N{}-ahead-{}.pdf".format(setting_name, N, T_ahead))
    plt.savefig("formal_figs/persistency-{}-N{}-ahead-{}.pdf".format(setting_name, N, T_ahead))
    plt.show()


if __name__ == "__main__":
    for setting_name in ["non-sa"]: #["random-size-10-dirichlet-0.05-({})".format(i) for i in [582, 355]] + ["new2-eight-states", "three-states", "non-sa", "non-sa-big2"]:
        visualize_ideal_action_persistency(setting_name)
