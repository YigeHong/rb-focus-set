import numpy as np

from rb_settings import *
from discrete_RB import *
from matplotlib import pyplot as plt
import os
from bisect import bisect



def compute_P_and_Phi_eigvals(setting):
    """
    return 1 if unichain and locally stable; return 0 if locally unstable;
    """
    analyzer = SingleArmAnalyzer(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, setting.suggest_act_frac)
    y = analyzer.solve_lp(verbose=False)[1]
    ind_neu = np.where(np.all([y[:,0] > analyzer.EPS, y[:,1] > analyzer.EPS],axis=0))[0]
    if len(ind_neu) > 1:
        return None

    Ppibs_centered = analyzer.Ppibs - np.outer(np.ones((analyzer.sspa_size,)), analyzer.state_probs)
    Ppibs_second_eig = np.max(np.abs(Ppibs_centered))

    Phi = analyzer.compute_Phi(verbose=False)
    Phi_spec_rad = np.max(np.abs(np.linalg.eigvals(Phi)))

    return analyzer, Ppibs_second_eig, Phi_spec_rad

def compute_mf_reward(setting, init_state_fracs, priority, T):
    mfrb = MeanFieldRB(setting.sspa_size, setting.trans_tensor, setting.reward_tensor, init_state_fracs)
    policy = PriorityPolicy(setting.sspa_size, priority, N=None, act_frac=setting.suggest_act_frac)

    total_reward = 0
    for t in range(T):
        cur_state_fracs = mfrb.get_state_fracs()
        sa_pair_fracs = policy.get_sa_pair_fracs(cur_state_fracs)
        instant_reward = mfrb.step(sa_pair_fracs)
        total_reward += instant_reward
    mfrb_avg_reward = total_reward / T

    return mfrb_avg_reward

def analyze_new_reward_modif(setting, direction):
    pass


def count_locally_unstable(alpha):
    sspa_size = 10
    distr = "dirichlet"
    laziness = None
    num_examples = 1000
    unichain_threshold = 0.95
    distr_and_parameter = distr
    if distr == "dirichlet":
        distr_and_parameter += "-" + str(alpha)
    if laziness is not None:
        distr_and_parameter += "-lazy-" + str(laziness)
    unichain_stability_table = np.zeros((2,2))
    eig_vals_list = []
    for i in range(num_examples):
        setting = RandomExample(sspa_size, distr, laziness=laziness, parameters=[alpha])
        result = compute_P_and_Phi_eigvals(setting)
        if result is not None:
            eig_vals = result[1:3]
            eig_vals_list.append(result[1:3])
            if eig_vals[0] < unichain_threshold:
                is_unichain = 1
            else:
                is_unichain = 0
            if eig_vals[1] < 1:
                is_locally_stable = 1
            else:
                is_locally_stable = 0
            unichain_stability_table[is_unichain, is_locally_stable] +=1
    print(unichain_stability_table)
    unstable_unichain_count = unichain_stability_table[1,0]
    unichain_count = np.sum(unichain_stability_table[1, :])
    eig_vals_list = np.array(eig_vals_list)
    plt.scatter(eig_vals_list[:,0], eig_vals_list[:,1])
    plt.plot([0,1], [1,1], linestyle="--", color="r")
    plt.plot([1,1], [0,1], linestyle="--", color="r")
    plt.xlabel("Second largest modulus of eigenvalues of P_pibs ")
    plt.ylabel("Spectral radius of Phi")
    plt.xlim([0, 1.1])
    plt.ylim([0, 2])
    plt.title("{}, unichain thresh {}, unstable / unichain = {} / {}".format(distr_and_parameter, unichain_threshold, unstable_unichain_count, unichain_count))
    plt.savefig("figs2/eigen-scatter-{}-size-{}.png".format(distr_and_parameter, sspa_size))
    plt.show()


def search_and_store_unstable_examples():
    # parameters
    num_examples = 10000
    num_reward_modif_examples = 0
    T_mf_simulation = 1000
    plot_subopt_cdf = False
    save_subopt_examples = False
    make_scatter_plot = False
    unichain_threshold = 0.95
    # hyperparameters
    sspa_size = 10
    distr = "dirichlet" #"uniform", "dirichlet", "CL
    laziness = None
    alpha = 0.1
    beta = 4 # > 3
    distr_and_parameter = distr
    if distr == "uniform":
        pass
    elif distr == "dirichlet":
        distr_and_parameter += "-" + str(alpha)
    elif distr == "CL":
        distr_and_parameter += "-" + str(beta)
    else:
        raise NotImplementedError
    if laziness is not None:
        distr_and_parameter += "-lazy-" + str(laziness)
    # check if exists: if so, load data; if not, create new
    if not os.path.exists("random_example_data"):
        os.mkdir("random_example_data")
    file_path = "random_example_data/random-{}-{}".format(sspa_size, distr_and_parameter)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            all_data = pickle.load(f)
        assert all_data["sspa_size"] == sspa_size
        assert all_data["distr"] == distr
        if distr == "dirichlet":
            assert all_data["alpha"] == alpha
        if distr == "CL":
            assert all_data["beta"] == beta
        assert all_data["laziness"] == laziness
        assert "examples" in all_data
    else:
        all_data = {}
        all_data["sspa_size"] = sspa_size
        all_data["distr"] = distr
        all_data["alpha"] = alpha
        all_data["beta"] = beta
        all_data["laziness"] = laziness
        all_data["examples"] = []
    # mode: search for new examples? add reward modification to existing example?
    num_exist_examples = len(all_data["examples"])
    if num_exist_examples < num_examples:
        if distr == "CL":
            B = ChungLuBeta2B(beta)
        # generate more examples if not enough
        for i in range(num_examples):
            if distr in ["uniform", "dirichlet"]:
                setting = RandomExample(sspa_size, distr, laziness=laziness, parameters=[alpha])
            elif distr == "CL":
                setting = ChungLuRandomExample(sspa_size, beta, B)
                print_bandit(setting)
            else:
                raise NotImplementedError
            if i > num_exist_examples:
                result = compute_P_and_Phi_eigvals(setting)
                if result is None:
                    all_data["examples"].append(None)
                    continue
                analyzer = result[0]
                setting.avg_reward_upper_bound = analyzer.opt_value
                setting.lp_priority = analyzer.solve_LP_Priority(verbose=False)
                setting.whittle_priority = analyzer.solve_whittles_policy()
                setting.unichain_eigval = result[1]
                setting.local_stab_eigval = result[2]
                print(setting.avg_reward_upper_bound, setting.lp_priority, setting.whittle_priority, setting.unichain_eigval, setting.local_stab_eigval)
                all_data["examples"].append(setting)
    # For the locally unstable examples, run the mean-field limit reward, and generate reward modification
    for i, setting in enumerate(all_data["examples"]):
        if setting is None:
            continue
        if (setting.local_stab_eigval > 1) and (setting.unichain_eigval < 1):
            if setting.avg_reward_lpp_mf_limit is None:
                print("analyzing the {}-th example's LP-Priority".format(i))
                # the subroutine of computing mf limit
                init_state_frac = np.ones((sspa_size,)) / sspa_size
                setting.avg_reward_lpp_mf_limit = compute_mf_reward(setting, init_state_frac, setting.lp_priority, T_mf_simulation)
            if setting.avg_reward_whittle_mf_limit is None:
                print("analyzing the {}-th example's Whittle index".format(i))
                init_state_frac = np.ones((sspa_size,)) / sspa_size
                if type(setting.whittle_priority) == int and (setting.whittle_priority < 0):
                    setting.avg_reward_whittle_mf_limit = 0
                else:
                    setting.avg_reward_whittle_mf_limit = compute_mf_reward(setting, init_state_frac, setting.whittle_priority, T_mf_simulation)
            if len(setting.reward_modifs) < num_reward_modif_examples:
                print("analyzing the reward modification of an unstable example, index={}".format(i))
                # subroutine of generating adversarial reward function (need to understand LP stability) and compute mean-field limit
                for i in range(num_reward_modif_examples):
                    direction = np.random.normal(0, 1, (sspa_size,))
                    direction = direction / np.linalg.norm(direction)
                    if i > len(setting.reward_modifs):
                        new_reward_modif = analyze_new_reward_modif(setting, direction)
                        setting.reward_modifs.append(new_reward_modif)

    if make_scatter_plot:
        eig_vals_list = []
        unstable_unichain_count = 0
        unichain_count = 0
        for i, setting in enumerate(all_data["examples"]):
            if setting is None:
                continue
            eig_vals_list.append([setting.unichain_eigval, setting.local_stab_eigval])
            if setting.unichain_eigval <= unichain_threshold:
                unichain_count +=1
                if setting.local_stab_eigval > 1:
                    unstable_unichain_count += 1
        eig_vals_list = np.array(eig_vals_list)
        colors = ["r" if eig_vals_list[i,1]>1 else "b" for i in range(eig_vals_list.shape[0])]
        plt.scatter(eig_vals_list[:,0], eig_vals_list[:,1], c=colors, s=1)
        plt.plot([0,1], [1,1], linestyle="--", color="r")
        plt.plot([1,1], [0,1], linestyle="--", color="r")
        plt.xlabel("Second largest modulus of eigenvalues of P_pibs ")
        plt.ylabel("Spectral radius of Phi")
        plt.xlim([0, 1.1])
        plt.ylim([0, 2])
        plt.title("{}, unichain thresh {}, unstable / unichain = {} / {}".format(distr_and_parameter, unichain_threshold, unstable_unichain_count, unichain_count))
        plt.savefig("figs2/eigen-scatter-{}-size-{}.png".format(distr_and_parameter, sspa_size))
        plt.show()

    subopt_ratios = []
    subopt_ratios_w = []
    for i, setting in enumerate(all_data["examples"]):
        if setting is None:
            continue
        if (setting.local_stab_eigval > 1) and (setting.unichain_eigval < 1):
            print("the {}-th example is locally unstable".format(i))
            subopt_ratio = setting.avg_reward_lpp_mf_limit / setting.avg_reward_upper_bound
            subopt_ratio_w = setting.avg_reward_whittle_mf_limit / setting.avg_reward_upper_bound
            subopt_ratios.append(subopt_ratio)
            subopt_ratios_w.append(subopt_ratio_w)
            if subopt_ratio < 0.9:
                print("In the {}-th example, lp index is {}-optimal, Whittle index is {}-suboptimal".format(i, subopt_ratio, subopt_ratio_w))
                setting_save_path =  "setting_data/random-size-{}-{}-({})".format(sspa_size, distr_and_parameter, i)
                # save suboptimal examples
                if save_subopt_examples:
                    print("saving the example...")
                    if os.path.exists(setting_save_path):
                        print(setting_save_path+" exists!")
                    else:
                        save_bandit(setting, setting_save_path, {"alpha":alpha})

    # visualize percentage of subopt examples
    if plot_subopt_cdf:
        name_data_dict = {"lpp":subopt_ratios, "whittle":subopt_ratios_w,
                          "max":[max(subopt_ratios[i], subopt_ratios_w[i]) for i in range(len(subopt_ratios))]}
        for name, data in name_data_dict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.8)
            data = np.sort(data)
            # print(bisect(data, 0.9) / len(data), bisect(data, 0.95) / len(data))
            ax.plot(data, np.linspace(0, 1, len(subopt_ratios)))
            ax.grid()
            # ax.hist(subopt_ratios, bins=20, weights=np.ones(len(subopt_ratios)) / len(subopt_ratios))
            if name == "lpp":
                full_name = "LP index"
            elif name == "whittle":
                full_name = "Whittle index"
            elif name == "max":
                full_name = "max of two indices"
            title=ax.set_title("Size-{}-{} \n Subopt of {}'s mean-field limit among {} non-UGAP examples\n ".format(sspa_size, distr_and_parameter, full_name, len(subopt_ratios)))
            fig.tight_layout()
            title.set_y(1.05)
            plt.savefig("figs2/nonugap-subopt-{}-size-{}-{}.png".format(name, sspa_size, distr_and_parameter))
            plt.show()

    print("saving data... do not quit...")
    with open(file_path, "wb") as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    np.random.seed(114514)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # for alpha in [0.05, 0.1, 0.2, 0.5, 1]:
        # count_locally_unstable(alpha)
    search_and_store_unstable_examples()
