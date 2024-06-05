import numpy as np
import cvxpy as cp
from discrete_RB import *
import rb_settings
import time


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
    policy = SetExpansionPolicy(setting.sspa_size, y, N, act_frac)

    cum_time = 0
    T = 5
    for i in range(T):
        tic = time.time()
        states = np.random.choice(sspa, N, replace=True)
        policy.get_new_focus_set(states=states, last_focus_set=np.array([], dtype=int))
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

    for num_terms in range(100, 1000, 100):
        W, spn_error = analyzer.compute_W(num_terms=num_terms)
    print("W={}, lamw <= {}, expand {} terms, spectral norm error<={}".format(W, np.linalg.norm(W, ord=2), num_terms, spn_error))


np.random.seed(114514)
# test_repeated_solver()
test_W_solver()


